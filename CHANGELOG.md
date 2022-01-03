# Changelog

## Added

- Comments have been added where applicable to better clarify the functionality.
- Folders to save the final resulting images will now be created automatically and named according to the input image folder name.

## Changed

- Absolute path for saving the final resulting images had been changed to relative path (#. Note that the `pytesseract.pytesseract.tesseract_cmd` still uses absolute path.

## Removed

- Removed processing time measurement and recording as I deemed it to be unreliable to measure actual image processing pipeline performance.
- Removed redundant `locate_license_plate()` functions inside `CannyANPR` and `EdgelessANPR` class in anprclass.py.

## Fixed

- Fixed an issue where the program still saves final resulting images despite debug (-d) being enabled.