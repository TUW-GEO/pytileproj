import shapely

from pytileproj.tile import RasterTile


class GeomOutOfZoneError(Exception):
    """Class to handle geometries outside projection zones."""

    def __init__(self, geom: shapely.Geometry) -> None:
        """Initialise a GeomOutOfZoneError.

        Parameters
        ----------
        geom: shapely.Geometry
            Geometry.

        """
        self.msg = (
            f"The given {geom.geom_type} ('{geom}') is "
            "not within the zone boundaries of the projection."
        )

    def __str__(self) -> str:
        """Return string representation of this class."""
        return self.msg


class TileOutOfZoneError(Exception):
    """Class to handle geometries outside projection zones."""

    def __init__(self, raster_tile: RasterTile) -> None:
        """Initialise a TileOutOfZoneError.

        Parameters
        ----------
        raster_tile: RasterTile
            Raster tile.

        """
        self.msg = (
            f"The given tile ('{raster_tile.boundary.geom}') is "
            "not within the zone boundaries of the projection."
        )

    def __str__(self) -> str:
        """Return string representation of this class."""
        return self.msg
