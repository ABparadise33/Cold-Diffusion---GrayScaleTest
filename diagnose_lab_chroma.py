"""Diagnose partial Lab chroma using an existing factor-1 Lab checkpoint."""

from gray_cold_diffusion.chroma_diagnostic import main


if __name__ == "__main__":
    main(color_space="lab")
