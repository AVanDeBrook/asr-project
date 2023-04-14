Data Processing and Analysis API
================================

Base Data Class
---------------

.. autoclass:: data.Data
    :show-inheritance:
    :members:
    :undoc-members:
    :member-order: bysource

Dataset-specific Classes
------------------------

These classes extend the base class ``Data`` and implement methods specific to the dataset being parsed.

The abstract function ``parse_transcripts`` from the ``Data`` class is implemented to parse all transcript
and audio data into a common format (NeMo manifest format or ``dict`` objects that can be dumped directly
to json strings).

.. autoclass:: data.ATCCompleteData
    :show-inheritance:
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: data.ATCO2SimData
    :show-inheritance:
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: data.ATCOSimData
    :show-inheritance:
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: data.ZCUATCDataset
    :show-inheritance:
    :members:
    :undoc-members:
    :member-order: bysource

Air-Traffic Control Complete Transcript Parsing Utility
-------------------------------------------------------

.. automodule:: data.atccutils
    :show-inheritance:
    :members:
    :member-order: bysource