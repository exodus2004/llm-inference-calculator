# LLM Inference Hardware Calculator

A web-based calculator to estimate hardware requirements for running Large Language Models (LLMs) in inference mode. This tool helps you determine the VRAM and system RAM needed for running different LLM configurations.

## Features

- Calculate VRAM requirements based on:
  - Model size (number of parameters)
  - Quantization method (FP32/FP16/INT8/INT4/etc.)
  - Context length
  - KV cache settings
- Support for both discrete GPUs and unified memory systems
- Estimates for:
  - Required VRAM
  - Minimum system RAM
  - On-disk model size
  - Number of GPUs needed

## Development

This project uses React + TypeScript + Vite.

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build
```

## Docker

- to use Docker and docker compose first create a `.env` file based on the [.env.example](.env.example) and set a PORT that should be exposed
- than run `docker compose up -d --build` to run the app

## Technical Notes

- Calculations are approximations and may vary based on specific implementations
- VRAM estimates include overhead for KV cache when enabled
- Unified memory calculations assume up to 75% of system RAM can be used as VRAM
- Discrete GPU calculations assume 24GB VRAM cards (like RTX 3090/4090)

## License

MIT
