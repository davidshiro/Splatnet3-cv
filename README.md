# Splatnet3-cv
 A project to take screenshots of Splatoon 3 Splatnet stats (only .jpg or .png!!!) and convert them to CSV.

 Install requirements: `sudo pip3 install -r requirements.txt`

 Install tesseract (Google's OCR engine): https://tesseract-ocr.github.io/tessdoc/Installation.html

 Additionally, ensure rosters.csv is populated with team data for name recognition to function.

 Your results will be output to output.xlsx.

 Command usage: `python match.py --images {folder where images are held}`

 Demo images in /images, run demo with command `python match.py --images images`

# To do
  - [X] Player info recognition
    - [X] Assists
    - [X] Name-team association algorithm
  - [ ] Match info recognition
    - [X] Mode
    - [X] Map
    - [ ] Timer
    - [ ] Score
  - [X] Conversion to .xlsx
  - [ ] Add redundant player detection

 Lower priority:

  - [ ] Rosters read from Excel 
  - [ ] Normalize image input format
  - [ ] Create GUI

# Credits
 Thanks to Leanny for providing weapon icons.
