#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import shlex
import os
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


def run_command(
    command: List[str],
    cwd: Optional[Union[str, os.PathLike]] = None,
    env: Optional[dict] = None,
    check: bool = True,
    capture_output: bool = True,
    text: bool = True
) -> CommandResult:
    """
    Führt ein Shell-Kommando sicher aus.
    Kein shell=True für maximale Sicherheit.
    """
    cmd_str = " ".join(shlex.quote(str(c)) for c in command)
    logger.debug(f"Executing command: {cmd_str}")

    try:
        proc = subprocess.run(
            [str(c) for c in command],
            cwd=cwd,
            env=env,
            capture_output=capture_output,
            text=text,
            check=False  # Wir handhaben das Fehlerhandling selbst
        )

        result = CommandResult(
            command=[str(c) for c in command],
            returncode=proc.returncode,
            stdout=proc.stdout if capture_output else "",
            stderr=proc.stderr if capture_output else ""
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
