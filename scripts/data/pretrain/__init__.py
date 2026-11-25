"""The pretraining data pipeline: upstream data at pinned revisions → training-ready leaves.

``pins`` holds every upstream, ``readers`` the per-dataset readers, ``common`` the stages'
helpers. ``scripts/data/pretrain_data.py`` is the command line.
"""
