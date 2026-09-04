"""Diagnose partial RGB color using an existing, explicitly checked RGB checkpoint."""

from gray_cold_diffusion.chroma_diagnostic import main


if __name__ == "__main__":
    main(color_space="rgb")
