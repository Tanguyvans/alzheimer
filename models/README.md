# Models Directory

Place your trained Alzheimer's prediction model weights here.

## Expected Format

- PyTorch: `alzheimer_model.pth`
- The model should be compatible with the `Simple3DCNN` architecture in `app/model.py`

## Example

```bash
cp /path/to/your/trained_model.pth ./alzheimer_model.pth
```

## Note

Model files are gitignored for security and size reasons. Never commit model weights to version control.
