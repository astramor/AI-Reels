# Offline model evaluation procedure

Use the same synthetic, non-sensitive German sermon fixture for every comparison:

- Fixture: `tests/fixtures/model_eval_sermon.txt`
- Format: `[HH:MM:SS --> HH:MM:SS] text`
- Only timestamps present in the fixture may be used. Fabricated timestamps are rejected by the offline evaluation check.

## Procedure

1. Select one provider/model with `check_models.py` or a CLI override.
2. Run the existing summary/highlight prompt against the fixture only.
3. Save the structured highlight JSON and generated titles/posts.
4. Evaluate offline:
   - JSON is valid and contains up to three highlights.
   - Every `start` and `end` timestamp originates from the fixture.
   - `start < end` and the span lies inside the fixture excerpt.
   - Valid non-overlapping highlights survive the existing `HighlightPayload` validation path.
   - Overlapping candidates follow the documented policy: higher-confidence candidate wins; finalized spans remain authoritative.
   - Fewer than three highlights are allowed when fewer valid distinct excerpts exist.
   - Each highlight is faithful to the source: no invented facts or unsupported claims.
   - German titles are understandable, specific, and free of forced or church-specific wording.
   - Latency and token usage are reported only when actually measured.
5. Use exact catalog model IDs. Do not treat labels containing `"max"` as API model identifiers or as proof of a supported reasoning parameter.
6. Send reasoning or structured-output parameters only when the selected model and existing client integration support them.
7. Do not infer model quality from catalog metadata alone.
8. Live inference comparisons require separate user approval and real credentials.

## Actual commands

List Gemini models:

```bash
.venv/bin/python check_models.py --provider gemini
```

List OpenRouter models:

```bash
.venv/bin/python check_models.py --provider openrouter
```

Search DeepSeek/Flash models:

```bash
.venv/bin/python check_models.py --provider openrouter --search deepseek --json
.venv/bin/python check_models.py --provider openrouter --search flash --json
```

Refresh the catalog:

```bash
.venv/bin/python check_models.py --provider openrouter --refresh
```

Select an exact model:

```bash
.venv/bin/python check_models.py --provider openrouter \
  --select deepseek/deepseek-v4.1-flash-1017
```

Select and save provider/model defaults:

```bash
.venv/bin/python check_models.py --provider openrouter \
  --select deepseek/deepseek-v4.1-flash-1017 --save
```

Run the pipeline with an explicit OpenRouter model:

```bash
OPENROUTER_API_KEY="$OPENROUTER_API_KEY" .venv/bin/python batch_reels.py \
  --raw_dir raw_videos \
  --llm-provider openrouter \
  --llm-model deepseek/deepseek-v4.1-flash-1017
```

## Credential isolation

OpenRouter uses only its own credential:

- explicit CLI `--openrouter-api-key`
- `OPENROUTER_API_KEY`
- `openrouter_api_key` or `openrouter.api_key` in `config.yaml`

`OPENAI_API_KEY` is not automatically reused for OpenRouter. The OpenAI provider continues to use `OPENAI_API_KEY` or `openai_api_key`.

Precedence is explicit CLI > environment > YAML fallback.

## Structured-output behavior

OpenRouter `response_format` is sent only when capability is known to be supported. The default `openrouter_json_mode` is `unknown`, so the parameter is omitted. Set it to `supported` only when catalog metadata confirms `response_format`; set it to `unsupported` when metadata says it is absent.

Unknown capability is treated as unsupported, not automatically supported. An explicit unsupported-parameter error may trigger one retry without `response_format`, using the same provider and model. Authentication, billing, and unrelated errors are not retried this way. Invalid JSON output raises a clear validation error.

## Live-test limitations

- OpenRouter live text-only comparison with `deepseek/deepseek-v4.1-flash-1017` succeeded and produced valid structured JSON.
- Gemini live text-only comparison with `models/gemini-3.5-flash` failed with `402 RESOURCE_EXHAUSTED` because prepayment credits were depleted. Gemini inference remains unverified.
- The previous un-timestamped fixture produced `0.0`/`30.0` excerpt-boundary values, which were not source timestamps. The fixture now contains explicit synthetic timestamped segments.
- Caches, live results, credentials, and personal configuration are excluded from the feature commit.

Catalog-only live check, requiring separate network approval and no inference:

```bash
.venv/bin/python check_models.py --provider openrouter \
  --search deepseek --json
```

Automated tests use mocked HTTP/SDK clients and fake credentials only.
