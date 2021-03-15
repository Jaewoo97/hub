.. Datature Hub documentation master file, created by
   sphinx-quickstart on Mon Mar 15 14:14:23 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Datature Hub
=========================================

Datature Hub is a loader for models trained on the Datature platform.

.. toctree::
   :maxdepth: 2
   :caption: Contents:


.. currentmodule:: dataturehub.hub

Model Types
-----------------------------------------

.. autoclass:: ModelType
   :show-inheritance:
   :members:

Downloading Model Weights
-----------------------------------------

You can use the :py:func:`download_model` function to download model weights
in your chosen format.

By default, models are saved in the folder returned by
:py:func:`get_default_hub_dir`, which is ``~/.dataturehub`` on MacOS
and Linux, and ``C:\Users\XXXX\.dataturehub`` on Windows.

.. autofunction:: get_default_hub_dir

.. autofunction:: download_model

Loading Models Directly from Datature Hub
-----------------------------------------

Datature Hub provides functions to download and immediately
instantiate machine learning models.

Models loaded via these functions are cached in the folder returned by
:py:func:`get_default_hub_dir`, or another folder if you specify the
``hub_dir`` parameter.

.. autofunction:: load_tf_model

   .. seealso::

      Documentation of :py:func:`tf.saved_model.load` for information
      about additional parameters that may be passed to this function.
      
