Installation
============

Requirements
-----------

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB RAM minimum

Dependencies
-----------

.. code-block:: bash

   # Core dependencies
   torch>=1.9.0
   transformers>=4.5.0
   numpy>=1.19.0
   scipy>=1.7.0

Installation Steps
---------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/yourusername/nivalde.git
      cd nivalde

2. Install dependencies:

   .. code-block:: bash

      pip install -r requirements.txt

3. Download pre-trained models:

   .. code-block:: bash

      python scripts/download_models.py

Configuration
------------

1. Set environment variables:

   .. code-block:: bash

      export NIVALDE_MODEL_PATH=/path/to/models
      export NIVALDE_DATA_PATH=/path/to/data

2. Configure model parameters in `config.yaml`
