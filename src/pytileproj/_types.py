from collections.abc import Generator, Iterator

from morecantile.models import Tile as RegularTile

from pytileproj.tile import IrregularTile, RasterTile

Extent = tuple[int | float, int | float, int | float, int | float]
PartialExtent = tuple[int | float, int | float, int | float | None, int | float | None]
AnyTile = RegularTile | IrregularTile
TileIterator = Iterator[AnyTile]
RegTileIterator = Iterator[RegularTile]
RasterTileGenerator = Generator[RasterTile, RasterTile, RasterTile]
