# %load_ext autoreload
# %autoreload 2

import torchvision
from medigan import Generators
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

# task 1
model_ids = [
    "00001_DCGAN_MMG_CALC_ROI",
    "00002_DCGAN_MMG_MASS_ROI",
    "00003_CYCLEGAN_MMG_DENSITY_FULL",
    "00004_PIX2PIX_MMG_MASSES_W_MASKS",
    "00019_PGGAN_CHEST_XRAY",
]

# Check number of models
assert len(model_ids) == 5, "Add another model"

# Check that each item in model_ids is valid
for m_id in model_ids:
    assert m_id in Generators().list_models()
else:
    print(f"All {len(model_ids)} models are valid!")

# task 2
import app

print("Models available in app:")
for model_id in app.model_ids:
    print(model_id)

