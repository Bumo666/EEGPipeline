class BaseBCIDataset:
    """Base interface for BCI dataset readers."""

    def load_raw(self):
        """Load raw EEG data for one dataset or subject."""

    def get_metadata(self):
        """Return dataset metadata such as channels, classes, and sampling rate."""

    def to_trials(self):
        """Convert raw EEG data into trial-level arrays and labels."""
