"""One module per reader or family. A module registers its datasets by defining ``PINS``
(``pins.Pin``) and ``RECIPES`` (``recipe.Recipe``) dicts keyed by registry dataset name;
``catalog`` collects them, so no shared table is edited."""
