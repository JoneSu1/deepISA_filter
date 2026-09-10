# Use a non-interactive matplotlib backend for the whole test suite so that
# plotting functions called without `outpath` (plt.show) never open windows
# or hit Tk errors on headless runs.
import matplotlib

matplotlib.use("Agg")
