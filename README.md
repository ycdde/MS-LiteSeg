# MS-LiteSeg

Official Pytorch implementations for MS-LiteSeg. 

For installation, dataset preparation, training, and evaluation, please refer to the [MMSegmentation official documentation](https://github.com/open-mmlab/mmsegmentation).  

## File Placement Guide  

Place the provided files in the following directories within your MMSegmentation project:

- **Config files** (`MS-LiteSeg_potsdam.py`, `MS-LiteSeg_vaihingen.py`) → `configs/msliteseg`
- **Neck** (`dfa.py`) → `mmseg/models/necks/`
- **Decoder** (`lightsegformer_head.py`) → `mmseg/models/decode_heads/`

After placing the files, update the corresponding `__init__.py` files in `mmseg/models/necks/` and `mmseg/models/decode_heads/` to import the new modules.

For usage, modify your configuration files to include the custom components.

---
For further details, please check the official [MMSegmentation documentation](https://github.com/open-mmlab/mmsegmentation).
