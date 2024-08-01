import cv2  
import easyocr  
import numpy as np 
from datetime import datetime  
import requests 
# import warnings  
import re  
from dateutil.relativedelta import relativedelta  
from difflib import SequenceMatcher 
import streamlit as st 

# Ignoring specific FutureWarning related to weights_only=False
# warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

# Initializing the EasyOCR reader with English language support and GPU processing
reader = easyocr.Reader(['en'], gpu=True)

# Function to format and sort dates
def format_and_sort_dates(dates):
    # Mapping of month abbreviations to numbers
    month_conversion = {
        "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
        "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
        "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"
    }
    
    formatted_dates = []
    for date in dates:
        # Removing non-alphanumeric characters
        date = re.sub(r'[^a-zA-Z0-9]', ' ', date)
        # Converting month abbreviations to numbers
        for month in month_conversion:
            date = re.sub(month, month_conversion[month], date, flags=re.IGNORECASE)
        # Reformatting the date string
        date = re.sub(r'\s+', '/', date).strip()
        formatted_dates.append(datetime.strptime(date, "%d/%m/%Y"))
    
    # Sorting the dates
    formatted_dates.sort()
    return formatted_dates

# Function to extract dates from OCR results
def extract_dates(results):
    month_abbrs = r'(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)'
    
    # Various date patterns to match
    date_patterns = [
        r'\b\d{2}/\d{2}/\d{4}\b',  # e.g., 31/12/2024
        r'\b\d{2}-\d{2}-\d{4}\b',  # e.g., 31-12-2024
        r'\b\d{4}-\d{2}-\d{2}\b',  # e.g., 2024-12-31
        r'\b\d{4}/\d{2}/\d{2}\b',  # e.g., 2024/12/31
        r'\b\d{2} ' + month_abbrs + r' \d{4}\b',  # e.g., 31 JAN 2024
        r'\b\d{2}-' + month_abbrs + r'-\d{4}\b',  # e.g., 31-JAN-2024
        r'\b\d{2}-' + month_abbrs + r'-\d{2}\b',  # e.g., 31-JAN-24
        r'\b\d{2}.\d{2}.\d{4}\b',  # e.g., 31.12.2024
        r'\b\d{2}.\d{2}.\d{2}\b',  # e.g., 31.12.24
        r'\b\d{2} \w{3} \w{3} \d{2}\b',  # e.g., 31 JAN JAN 24
        r'\b\d{2} ' + month_abbrs + ' ' + month_abbrs + r' \d{2}\b',  # e.g., 31 JAN /JAN 24
        r'\b\d{2} ' + month_abbrs + '/' + month_abbrs + r' \d{2}\b',  # e.g., 31 JAN/JAN 24
        r'\b\d{2} ' + month_abbrs + r' \d{2}\b',  # e.g., 31 JAN 24
    ]
    
    dates = set()
    
    # Searching for date patterns in OCR results
    for (_, text, _) in results:
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                dates.add(match)
    
    return list(dates)

# Function to correct date using month search
def correct_date_with_month_search(date_str, text_lines, date_type):
    # Mapping of month abbreviations to numbers
    month_conversion = {
        "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
        "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
        "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"
    }
    
    months_found = set()
    # Searching for months in the text lines
    for _, line, _ in text_lines:
        line_upper = line.upper()
        for month in month_conversion:
            if re.search(r'\b' + month + r'\b', line_upper):
                months_found.add(month)
    
    # If enough months are found, correct the date string
    if len(months_found) >= 3:
        if date_type == 'birth':
            month = months_found[0]
        elif date_type == 'expiry':
            month = months_found[-1]
        corrected_date = re.sub(r'[^0-9]', '', date_str)
        corrected_date = corrected_date[:2] + month_conversion[month] + corrected_date[4:]
    else:
        # Apply character corrections if months are not found
        corrections = {
            'A': '1', 'B': '8', 'C': '0', 'D': '0', 'E': '3', 'F': '7', 'G': '6',
            'H': '4', 'I': '1', 'J': '1', 'K': '1', 'L': '1', 'M': '0', 'N': '0',
            'O': '0', 'P': '9', 'Q': '0', 'R': '2', 'S': '5', 'T': '7', 'U': '0',
            'V': '8', 'W': '8', 'X': '8', 'Y': '4', 'Z': '2'
        }
        corrected_date = ''.join(corrections.get(c, c) for c in date_str)
    
    return corrected_date

# Function to correct passport number
def correct_passport_number(passport_number):
    corrections = {
        'O': '0',
        'I': '1',
        'B': '8',
        'G': '4'
    }
    corrected_passport_number = list(passport_number)
    # Applying character corrections to the passport number
    for i, char in enumerate(corrected_passport_number):
        if i > 0 and char in corrections:
            corrected_passport_number[i] = corrections[char]
    return ''.join(corrected_passport_number)

# Function to fetch country data
def fetch_data():
    url = "https://restcountries.com/v3.1/all"
    response = requests.get(url)
    data = response.json()
    
    icao_codes = set()
    for country in data:
        if "cca3" in country:  
            icao_codes.add(country["cca3"].upper())

    country_code_to_nationality = {}
    for country in data:
        if "cca3" in country and "name" in country:
            cca3 = country["cca3"].upper()
            nationality = country["name"]["official"].capitalize()
            country_code_to_nationality[cca3] = nationality
    
    return icao_codes, country_code_to_nationality

# Function to calculate Levenshtein distance
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

# Function to correct nationality based on text lines
def correct_nationality(nationality, text_lines):
    corrections = {
        '0': 'O',
        '1': 'I',
        '2': 'Z',
        '5': 'S',
        '8': 'B'
    }
    corrected_nationality = list(nationality)
    # Applying character corrections to the nationality
    for i, char in enumerate(corrected_nationality):
        if char in corrections:
            corrected_nationality[i] = corrections[char]
    corrected_nationality_str = ''.join(corrected_nationality)

    valid_country_codes, _ = fetch_data()
    
    if corrected_nationality_str not in valid_country_codes:
        all_texts = []
        # Collecting all text lines for comparison
        for _, line, _ in text_lines:
            all_texts.extend(line.split())
        
        min_score = float('inf')
        closest_match = corrected_nationality_str
        for text in all_texts:
            score = levenshtein_distance(corrected_nationality_str, text.upper())
            if score < min_score:
                min_score = score
                closest_match = text.upper()
        
        return closest_match
    
    return corrected_nationality_str

# Function to preprocess the image for OCR
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    
    # Resizing the image
    scale_percent = 200 
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
    
    # Converting the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Applying Gaussian blur to the image
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Applying adaptive thresholding to the image
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 5)
    
    return binary

# Function to extract text from the image using OCR
def extract_text(image_path):
    image = preprocess_image(image_path)
    results = reader.readtext(image)
    
    mrz_lines = []
    # Extracting lines of text that are likely part of the MRZ (longer texts)
    for (bbox, text, prob) in results:
        if len(text) >= 15:  
            mrz_lines.append(text)
    
    return results, mrz_lines[-2:], image

# Function to parse MRZ lines to extract passport fields
def parse_mrz(mrz_lines, text_lines):
    fields = {}
    # Cleaning MRZ lines to keep only alphanumeric characters and '<'
    mrz_lines = [''.join(c if c.isalnum() or c == '<' else '' for c in line) for line in mrz_lines]
    
    # Extracting name information from the MRZ
    if '<<' in mrz_lines[0]:
        name_start_index = mrz_lines[0].find('<<') + 2
    else:
        name_start_index = mrz_lines[0].find('<') + 1
    
    name_data_raw = mrz_lines[0][name_start_index:44]
    name_data = [entry for entry in name_data_raw.split('<') if entry and len(entry) > 1]
    fields['name'] = ' '.join(name_data).replace('<', ' ').strip().upper()
    
    # Extracting and formatting date of birth
    date_of_birth_raw = mrz_lines[1][13:19]
    date_of_birth_corrected = correct_date_with_month_search(date_of_birth_raw, text_lines, 'birth')
    date_of_birth_formatted = f"{date_of_birth_corrected[4:6]}/{date_of_birth_corrected[2:4]}/19{date_of_birth_corrected[0:2]}"
    fields['date_of_birth'] = date_of_birth_formatted

    # Extracting and formatting date of expiry
    date_of_expiry_raw = mrz_lines[1][21:27]
    date_of_expiry_corrected = correct_date_with_month_search(date_of_expiry_raw, text_lines, 'expiry')
    date_of_expiry_formatted = f"{date_of_expiry_corrected[4:6]}/{date_of_expiry_corrected[2:4]}/20{date_of_expiry_corrected[0:2]}"
    fields['date_of_expiry'] = date_of_expiry_formatted

    # Fetching nationality information
    _, country_code_to_nationality = fetch_data()
    nationality = mrz_lines[1][10:13].upper()
    corrected_nationality = correct_nationality(nationality, text_lines)
    fields['nationality'] = country_code_to_nationality.get(corrected_nationality, corrected_nationality).upper()

    # Extracting passport type
    fields['passport_type'] = re.sub('[^A-Za-z]', '', mrz_lines[0][0:2]).upper()
    
    # Extracting and correcting passport number
    passport_number = mrz_lines[1][0:9].replace('<', '').upper()
    fields['passport_number'] = correct_passport_number(passport_number)
    fields['date_of_issue'] = ''
    fields['authority'] = ''

    return fields

# Function to calculate sequence similarity between two strings
def sequence_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Function to extract issuing authority from OCR results
def extract_authority(results):
    # List of keywords related to issuing authority
    authority_keywords = ['Place of Issue', 'Authority', 'Place of Issuance', 'Issued by', 'Issuing Office', 'Code of Issuing', 'Issuing Country', 'Issuing Authority']
    max_similarity = 0
    authority_bbox = None
    authority_text = ""

    # Searching for the most similar text to authority keywords
    for (bbox, text, prob) in results:
        for keyword in authority_keywords:
            similarity = sequence_similarity(text.lower(), keyword.lower())
            if similarity > max_similarity:
                max_similarity = similarity
                authority_bbox = bbox
                authority_text = text
    
    if authority_bbox is None:
        return "Authority not found", None
    
    return authority_text, authority_bbox

# Function to generate points on the edges of a bounding box
def generate_points_on_edges(bbox, num_points=10):
    points = []

    for i in range(len(bbox)):
        start = np.array(bbox[i])
        end = np.array(bbox[(i + 1) % len(bbox)])
        edge_points = [start + t * (end - start) for t in np.linspace(0, 1, num_points)]
        points.extend(edge_points)

    return points

# Function to find the closest text to a given bounding box
def find_closest_text(bbox, results, exclude_bbox):
    min_distance = float('inf')
    closest_text = ""

    bbox_points = generate_points_on_edges(bbox)

    for (other_bbox, text, prob) in results:
        if other_bbox == exclude_bbox or prob <= 0.5:
            continue

        other_bbox_points = generate_points_on_edges(other_bbox)

        current_min_distance = float('inf')

        for bbox_point in bbox_points:
            for other_bbox_point in other_bbox_points:
                distance = np.linalg.norm(bbox_point - other_bbox_point)
                if distance < current_min_distance:
                    current_min_distance = distance

        if current_min_distance < min_distance:
            min_distance = current_min_distance
            closest_text = text

    return closest_text

# Main function to process the image and extract relevant fields
def main(image_path):
    results, mrz_lines, _ = extract_text(image_path)

    # If MRZ lines are detected, parse them to extract fields
    if len(mrz_lines) >= 2:  
        fields = parse_mrz(mrz_lines[:2], results) 
    else:
        print("MRZ not fully detected")
        fields = {}

    dates = extract_dates(results)
    formatted_dates = format_and_sort_dates(dates)

    # Handling different cases for date fields
    if len(formatted_dates) == 3:
        fields['date_of_issue'] = formatted_dates[1].strftime("%d/%m/%Y")
        fields['date_of_birth'] = formatted_dates[0].strftime("%d/%m/%Y")
        fields['date_of_expiry'] = formatted_dates[2].strftime("%d/%m/%Y")
    else:
        if 'date_of_issue' not in fields or not fields['date_of_issue']:
            if 'date_of_expiry' in fields:
                try:
                    date_of_expiry_date = datetime.strptime(fields['date_of_expiry'], "%d/%m/%Y")
                    date_of_issue_date = date_of_expiry_date - relativedelta(years=10)
                    fields['date_of_issue'] = date_of_issue_date.strftime("%d/%m/%Y")
                except ValueError as e:
                    print(f"Date parsing error: {e}")
                    fields['date_of_issue'] = '00/00/00'

    _, authority_bbox = extract_authority(results)
    
    if authority_bbox:
        authority_value = find_closest_text(authority_bbox, results, authority_bbox)
        fields['authority'] = authority_value

    output_data = []

    # Preparing data for output
    for (bbox, text, prob) in results:
        bbox = [list(map(int, point)) for point in bbox]
        output_data.append({
            "text": text,
            "bounding_box": bbox
        })

    return fields, output_data

# Function to set custom style for Streamlit app
def set_style():
    st.markdown("""
        <style>
        .css-1w3yfkh {
            background-color: #0066cc;
            color: #ffffff;
            text-align: center;
            font-size: 1rem;
            font-weight: bold;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)

# Main Streamlit app function
def main_app():
    set_style()
    st.markdown('<div class="css-1w3yfkh">By Aida Gomezbueno for AKW Consultants</div>', unsafe_allow_html=True)
    st.title("OCR Verification Tool")

    # File uploader for image input
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner('Processing...'):
            fields, output_data = main(temp_image_path)
        
        st.image(temp_image_path, caption='Analyzed Passport', use_column_width=True)
        st.json(fields)

if __name__ == "__main__":
    main_app()