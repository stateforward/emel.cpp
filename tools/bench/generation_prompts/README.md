# Generation Prompt Fixtures

These files pin the maintained generation prompt inputs used by the compare bench.

## Contract

- `schema`: prompt fixture schema version
- `id`: stable prompt fixture identifier
- `shape`: prompt shape the bench knows how to render
- `text`: literal prompt payload
- `prompt_id`: canonical prompt identity echoed into compare records

## Current Shapes

- `single_user_text_v1`: render one user message with the checked-in text payload
