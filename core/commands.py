#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import shlex
import os
import signal
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Union
import logging

# Standard Logger konfigurieren
logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """Hält das Ergebnis eines ausgeführten Shell-Kommandos."""
    command: List[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.returncode == 0


class CommandError(Exception):
    """Exception für fehlgeschlagene Kommandos."""
    def __init__(self, result: CommandResult):
        self.result = result
        super().__init__(
            f"Command failed with exit code {result.returncode}: "
            f"{' '.join(shlex.quote(c) for c in result.command)}\n"
            f"Stderr: {result.stderr.strip()}"
        )


class ProcessRegistry:
    """Trackt aktive Subprozesse für sauberes Beenden."""
    _processes = set()

    @classmethod
    def register(cls, proc: subprocess.Popen):
        cls._processes.add(proc)

    @classmethod
    def unregister(cls, proc: subprocess.Popen):
        cls._processes.discard(proc)

    @classmethod
    def terminate_all(cls):
        if not cls._processes:
            return
        
        logger.warning(f"Terminating {len(cls._processes)} registered processes...")
        for proc in list(cls._processes):
            try:
                # Versuche die ganze Prozessgruppe zu beenden
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
        cls._processes.clear()


def _get_cuda_env(base_env: Optional[dict] = None) -> dict:
    """
    Erstellt ein Environment mit erweiterten LD_LIBRARY_PATH für NVIDIA Bibliotheken im venv.
    """
    env = (base_env or os.environ).copy()
    
    # Pfad zum aktuellen venv finden
    venv_path = os.environ.get("VIRTUAL_ENV")
    if not venv_path:
        # Fallback: Versuche relativ zum Script-Pfad
        script_dir = Path(__file__).parent.parent
        venv_path = str(script_dir / ".venv")
    
    if os.path.exists(venv_path):
        # Alle nvidia/*/lib Verzeichnisse finden
        lib_paths = []
        site_packages = Path(venv_path) / "lib"
        if site_packages.exists():
            # Suche nach nvidia und triton lib Verzeichnissen
            for p in site_packages.glob("python*/site-packages/nvidia/*/lib"):
                lib_paths.append(str(p))
            for p in site_packages.glob("python*/site-packages/triton/backends/nvidia/lib"):
                lib_paths.append(str(p))
        
        if lib_paths:
            existing_ld_path = env.get("LD_LIBRARY_PATH", "")
            new_ld_path = ":".join(lib_paths)
            if existing_ld_path:
                new_ld_path = f"{new_ld_path}:{existing_ld_path}"
            env["LD_LIBRARY_PATH"] = new_ld_path
            logger.debug(f"Extended LD_LIBRARY_PATH with {len(lib_paths)} NVIDIA paths")
            
    return env


def run_command(
    command: List[str],
    cwd: Optional[Union[str, os.PathLike]] = None,
    env: Optional[dict] = None,
    check: bool = True,
    capture_output: bool = True,
    text: bool = True
) -> CommandResult:
    """
    Führt ein Shell-Kommando sicher aus und registriert es in der ProcessRegistry.
    Inkludiert automatisch CUDA-Bibliothekspfade aus dem venv.
    """
    cmd_str = " ".join(shlex.quote(str(c)) for c in command)
    logger.debug(f"Executing command: {cmd_str}")

    # Automatische CUDA Environment-Anpassung
    full_env = _get_cuda_env(env)

    try:
        # Wir nutzen Popen statt run, um den Prozess während der Ausführung tracken zu können
        proc = subprocess.Popen(
            [str(c) for c in command],
            cwd=cwd,
            env=full_env,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            text=text,
            start_new_session=True # Prozessgruppe trennen für besseres Signal-Handling
        )
        
        ProcessRegistry.register(proc)
        
        try:
            stdout, stderr = proc.communicate()
        finally:
            ProcessRegistry.unregister(proc)

        result = CommandResult(
            command=[str(c) for c in command],
            returncode=proc.returncode,
            stdout=stdout if capture_output else "",
            stderr=stderr if capture_output else ""
        )

        if check and not result.success:
            raise CommandError(result)

        return result

    except Exception as e:
        if isinstance(e, CommandError):
            raise e
        
        # Für andere Fehler (z.B. Datei nicht gefunden)
        logger.error(f"Failed to execute command '{cmd_str}': {e}")
        raise
