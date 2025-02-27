def patch_time_series_decoder():
    """
    Monkey patch the TimeSeriesDecoder class to expose categorical dimensions
    """
    from hephaestus.models.models import TimeSeriesDecoder

    # Add a property to access categorical dimensions
    def categorical_dims_property(self):
        """Return the number of classes for each categorical variable"""
        if not hasattr(self, "_categorical_output_dims"):
            return [41]  # Default value if not properly initialized
        return self._categorical_output_dims

    # Apply the patch
    setattr(TimeSeriesDecoder, "categorical_dims", property(categorical_dims_property))

    # Save original __init__ method
    original_init = TimeSeriesDecoder.__init__

    # Create new init method that saves categorical dimensions
    def new_init(self, config, *args, **kwargs):
        # Call original init
        original_init(self, config, *args, **kwargs)

        # Store categorical dimensions
        if hasattr(config, "categorical_cardinality"):
            self._categorical_output_dims = config.categorical_cardinality
        else:
            print(
                "Warning: TimeSeriesConfig doesn't have categorical_cardinality. Using default value."
            )
            self._categorical_output_dims = [41]  # Default fallback

    # Replace init method
    TimeSeriesDecoder.__init__ = new_init
    TimeSeriesDecoder.__init__ = new_init
