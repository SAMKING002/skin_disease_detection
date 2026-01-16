import os
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, session
from werkzeug.utils import secure_filename
from flask_mysqldb import MySQL
from flask_bcrypt import Bcrypt
from datetime import datetime
import MySQLdb.cursors
app = Flask(__name__)
from flask import g

@app.teardown_appcontext
def close_db_connection(exception=None):
    """Properly close MySQL connection to avoid '2006 MySQL has gone away' errors."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


# Directory to store uploaded images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# directory for storing images being used by the signed user

UPLOAD_FOLDERS = 'static/uploads'
app.config['UPLOAD_FOLDERS'] = UPLOAD_FOLDERS
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load disease classification model
MODEL_PATH = "mobilenetv2_skins_model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(" Model loaded successfully.")
except Exception as e:
    print(f" Error loading model: {e}")

# Define class labels
CLASS_LABELS = [
    "Acne & Rosacea", "Actinic Keratosis & Malignant Lesions", "Atopic Dermatitis", "Bullous Disease",
    "Bacterial Infections", "Eczema", "Exanthems & Drug Eruptions", "Hair Loss & Hair Diseases",
    "Herpes, HPV & STDs", "Light Disorders", "Lupus & Connective Tissue Diseases", "Melanoma & Moles",
    "Nail Fungus & Nail Diseases", "Poison Ivy & Contact Dermatitis", "Psoriasis & Related Diseases",
    "Scabies & Infestations", "Seborrheic Keratoses & Benign Tumors", "Systemic Disease", "Fungal Infections",
    "Urticaria (Hives)", "Vascular Tumors", "Vasculitis", "Warts & Viral Infections"
]


# Define disease descriptions
DISEASE_DESCRIPTIONS = {
    "Acne & Rosacea": """**Acne & Rosacea: Understanding the Differences**
    
Both acne and rosacea cause skin breakouts, redness, and inflammation, but they are different conditions with different causes and treatments.

1. Acne (Pimples & Breakouts)
What is Acne?  
Acne is a common skin condition caused by clogged hair follicles due to oil (sebum), dead skin cells, and bacteria. It leads to pimples, blackheads, whiteheads, and cysts on the face, chest, and back.

Causes of Acne:  
Hormonal changes (puberty, pregnancy, menstrual cycles).
Excess oil (sebum) production.
Bacteria (Propionibacterium acnes).
Clogged pores due to dead skin cells.
Diet (high sugar and dairy may worsen acne).
Stress (can trigger breakouts).

Types of Acne:  
Blackheads & Whiteheads: Small clogged pores (black = open, white = closed).
Papules & Pustules: Red, swollen bumps (pustules contain pus).
Cystic Acne: Deep, painful lumps filled with pus (can cause scars).

How to Treat Acne?  
Mild Acne: Use gentle face wash, salicylic acid, or benzoyl peroxide.
Moderate Acne: Apply topical antibiotics (clindamycin) or retinoids.
Severe Acne: May need oral antibiotics or isotretinoin (under doctor’s advice).

Lifestyle Tips:
Wash face twice daily with a mild cleanser.
Avoid touching or popping pimples.
Eat a healthy diet and drink plenty of water.
Manage stress and get enough sleep.

2. Rosacea (Chronic Redness & Sensitivity)
What is Rosacea?
Rosacea is a long-term skin condition that causes redness, visible blood vessels, and sometimes pimples (similar to acne) on the face. It mostly affects the cheeks, nose, forehead, and chin.

Causes of Rosacea:  
Unknown exact cause (but linked to genetics and immune system).
Inflammation & overactive blood vessels.
Demodex mites (tiny skin mites that can trigger inflammation).
Triggers like sun, heat, spicy foods, alcohol, stress, and skincare products.

Types of Rosacea:  
Erythematotelangiectatic Rosacea (ETR): Redness, visible veins, and flushing.
Papulopustular Rosacea: Pimples and swelling (often mistaken for acne).
Phymatous Rosacea: Thickened, bumpy skin (commonly on the nose).
Ocular Rosacea: Red, irritated eyes and swollen eyelids.

How to Treat Rosacea? 
Gentle skincare (avoid harsh products and scrubbing).
Sunscreen (SPF 30+) is essential to protect sensitive skin.
Avoid triggers (spicy foods, alcohol, extreme temperatures).

Medications:
Metronidazole or azelaic acid creams for redness.
Oral antibiotics (doxycycline) for severe cases.
Laser therapy for visible blood vessels.
""",
    
 "Actinic Keratosis & Malignant Lesions": """Actinic Keratosis & Malignant Lesions: What You Need to Know

Both Actinic Keratosis (AK) and Malignant Lesions (Skin Cancer) are caused by sun damage and can affect the skin over time. AK is a warning sign, while malignant lesions mean cancerous skin growths.

1. Actinic Keratosis (AK) – A Warning Sign
What is Actinic Keratosis?
Actinic Keratosis (also called solar keratosis) is a precancerous skin condition caused by long-term sun exposure. These rough, scaly patches develop on sun-exposed areas like the face, ears, scalp, hands, and arms. If left untreated, AK can turn into skin cancer (squamous cell carcinoma).

Causes of Actinic Keratosis:  
Chronic sun exposure (UV radiation from the sun or tanning beds).
Fair skin, light hair, and blue or green eyes (higher risk).
Weak immune system (due to medical conditions or treatments).

Symptoms of Actinic Keratosis:
Small, rough, dry, or scaly patches (may feel like sandpaper).
Pink, red, or brownish spots (can blend into the skin).
May itch, burn, or feel sensitive.
Common on sun-exposed areas (face, scalp, hands, arms, ears, lips).

How to Treat Actinic Keratosis? 
Cryotherapy (freezing the spot with liquid nitrogen).
Topical creams (like 5-fluorouracil or imiquimod) to remove damaged cells.
Laser therapy or photodynamic therapy to destroy abnormal skin cells.
Regular skin checks to prevent progression to skin cancer.

2. Malignant Lesions (Skin Cancer)
What are Malignant Lesions?  
Malignant lesions are cancerous skin growths that develop when skin cells grow abnormally due to DNA damage (mostly from the sun). The three main types of skin cancer are:

Basal Cell Carcinoma (BCC): Most common, slow-growing, rarely spreads.
Squamous Cell Carcinoma (SCC): Can spread if untreated, often starts as AK.
Melanoma: The deadliest type, starts as an irregular mole.

How to Detect & Treat Malignant Lesions?  
Use the ABCDE Rule (Asymmetry, Border, Color, Diameter, Evolving).
Treatments include surgical removal, Mohs surgery, radiation, chemotherapy.

Prevention: 
Use SPF 30+ sunscreen, avoid tanning beds, wear protective clothing.
Regular dermatologist visits and self-exams.
""",
"Atopic Dermatitis (Eczema)": """Atopic Dermatitis (Eczema)
    
What is Atopic Dermatitis?
Atopic Dermatitis (AD), commonly known as eczema, is a chronic inflammatory skin disorder that causes intense itching, dryness, and red rashes. It often starts in childhood but can persist into adulthood. The condition is linked to genetics, immune system dysfunction, and environmental factors, making the skin more vulnerable to irritants and allergens.

Causes & Risk Factors
Atopic Dermatitis occurs due to a combination of:
Genetics: Family history of eczema, asthma, or hay fever increases the risk.
Immune system dysfunction: Overreaction to environmental triggers leads to inflammation.
Skin barrier defects: The skin fails to retain moisture, causing dryness and irritation.

Environmental Triggers:
Allergens (pollen, pet dander, dust mites).
Harsh soaps, detergents, and fragrances.
Climate changes (cold, dry air).
Stress and anxiety.
Certain foods (dairy, eggs, nuts, soy in some cases).

Symptoms
Severe itching (pruritus), often worse at night.
Dry, scaly, or thickened skin with red or brownish-gray patches (common on the hands, feet, ankles, wrists, neck, eyelids, inside elbows, and behind the knees).
Cracking, weeping, or crusting skin (in severe cases, open sores develop, increasing the risk of infection).
Recurring flare-ups, with periods of worsening symptoms.

Types of Atopic Dermatitis
Infantile Eczema: Common in babies, affecting the face, scalp, and arms.
Childhood Atopic Dermatitis: Typically appears on the elbows, knees, and hands.
Adult Atopic Dermatitis: Chronic, widespread, and often affects the neck and hands.

How to Treat Atopic Dermatitis?
Since there is no cure, treatment focuses on managing symptoms and preventing flare-ups:

1. Skincare & Moisturization
Use fragrance-free, hypoallergenic moisturizers (such as petroleum jelly, ceramide-based creams) at least **twice a day.
Take lukewarm baths, avoid long hot showers.
Use gentle, sulfate-free, fragrance-free cleansers.

2. Medications & Prescription Treatments
Topical corticosteroids (Hydrocortisone, Clobetasol): Reduce redness and swelling.
Calcineurin inhibitors (Tacrolimus, Pimecrolimus): Used for sensitive areas like the face and eyelids.
Oral antihistamines (Diphenhydramine, Cetirizine): Help control itching.
Immunosuppressants (Cyclosporine, Methotrexate, Dupilumab): For severe cases unresponsive to other treatments.
Antibiotics (Cephalexin, Dicloxacillin): If bacterial infection occurs due to scratching.

3. Lifestyle & Home Remedies
Wear soft, breathable cotton clothing (avoid wool & synthetic fabrics).
Keep nails short and clean to prevent scratching.
Avoid triggers like fragranced products, dust mites, and pet dander.
Use a humidifier in dry climates to maintain skin moisture.
Dietary changes (some people benefit from avoiding dairy, eggs, or gluten).

4. Emerging Treatments & Therapies
Biologic medications (Dupilumab – Dupixent®): Helps moderate-to-severe eczema cases.
Phototherapy (UV light therapy): Helps control symptoms when other treatments fail.
""",

    "Bullous Diseases": """Bullous Diseases (Autoimmune Blistering Disorders)

What are Bullous Diseases?  
Bullous diseases are a group of rare autoimmune skin conditions that cause painful, fluid-filled blisters on the skin and mucous membranes. These disorders occur when the immune system mistakenly attacks proteins in the skin, leading to skin separation and blister formation.

Common Types of Bullous Diseases  
- Bullous Pemphigoid (BP): Most common, affects older adults, causes tense blisters.
- Pemphigus Vulgaris (PV): Affects the mucous membranes (mouth, throat), causing fragile blisters that break easily.
- Dermatitis Herpetiformis (DH): Associated with celiac disease, causes itchy blisters on the elbows, knees, and buttocks.
- Linear IgA Disease (LAD): Rare autoimmune blistering disorder seen in both adults and children.

Causes & Risk Factors  
- Autoimmune response: Body mistakenly attacks skin proteins.
- Certain medications: Penicillins, NSAIDs, diuretics, and ACE inhibitors can trigger bullous diseases.
- Underlying conditions: Some cases are linked to cancer, inflammatory bowel disease, or infections.

Symptoms  
- Large, fluid-filled blisters that are tense or fragile.
- Itchy, red, inflamed skin before blisters appear.
- Painful mouth sores (Pemphigus Vulgaris).
- Crusting and scarring as blisters heal.

How to Treat Bullous Diseases?  
1. Medications  
- Corticosteroids (Prednisone, Clobetasol): Reduce inflammation and prevent new blisters.
- Immunosuppressants (Azathioprine, Mycophenolate Mofetil): Help suppress the immune attack.
- Biologic therapies (Rituximab): Target immune cells responsible for blister formation.
- Dapsone (for Dermatitis Herpetiformis): An antibiotic that controls itching and inflammation.

2. Skin Care & Prevention  
- Avoid scratching or popping blisters to reduce infection risk.
- Use non-stick dressings to cover affected areas.
- Gentle skincare – Avoid harsh soaps, use fragrance-free moisturizers.

3. Dietary Considerations (for Dermatitis Herpetiformis)  
- Strict gluten-free diet can dramatically reduce symptoms.
""",
"Bacterial Infections": """**Bacterial Infections of the Skin: Causes, Symptoms, and Treatments**

What are Bacterial Skin Infections?  
Bacterial skin infections occur when harmful bacteria invade the skin, leading to redness, swelling, pain, and sometimes pus formation. These infections range from mild to severe, with some requiring antibiotics or medical intervention.

Common Types of Bacterial Skin Infections  
1. Impetigo  
   - A contagious infection causing red sores and honey-colored crusts.
   - Common in children and often appears around the nose, mouth, and hands.
   - Caused by: Staphylococcus aureus or Streptococcus pyogenes.
   - Treatment: Topical or oral antibiotics like mupirocin or cephalexin.

2. Cellulitis  
   - A deeper skin infection causing painful swelling, warmth, and redness.
   - Can spread quickly and cause fever, chills, or lymph node swelling.
   - Caused by: Streptococcus or Staphylococcus bacteria entering a break in the skin.
   - Treatment: Oral or IV antibiotics (penicillin, clindamycin).

3. Erysipelas  
   - A more superficial and well-defined form of cellulitis with raised, red, swollen patches.
   - Often affects the face or legs.
   - Caused by: Streptococcus pyogenes.
   - Treatment: Penicillin or cephalosporins.

4. Boils (Furuncles) and Carbuncles  
   - Boils: Deep, pus-filled infections of hair follicles.
   - Carbuncles: Clusters of boils that extend deeper.
   - Caused by: Staphylococcus aureus, including MRSA (Methicillin-Resistant Staphylococcus aureus).
   - Treatment: Warm compresses, incision and drainage, antibiotics for severe cases.

5. Necrotizing Fasciitis (Flesh-Eating Disease) – Severe Case  
   - A rare but life-threatening infection destroying deep tissue layers.
   - Symptoms: Intense pain, rapid swelling, skin discoloration, blisters, fever, and shock.
   - Caused by: Group A Streptococcus or Clostridium bacteria.
   - Treatment: Emergency surgery, IV antibiotics, and supportive care.

Prevention Tips  
- Keep wounds clean and covered.
- Avoid sharing personal items (razors, towels).
- Wash hands frequently.
- Treat cuts, bites, and scrapes promptly.

When to See a Doctor?  
Seek medical help if you experience:
- Spreading redness or streaking.
- Fever or chills.
- Increasing pain, warmth, or pus.
- Rapid skin deterioration.

""",

    "Eczema": """**Eczema (Atopic Dermatitis): Causes, Symptoms, and Treatment**

### What is Eczema?
Eczema (Atopic Dermatitis) is a **chronic inflammatory skin condition** that causes **intense itching, redness, dryness, and rashes**. It often begins in childhood but can affect people of all ages.

### Causes & Risk Factors
Eczema is caused by a combination of:
- **Genetics:** Family history of eczema, asthma, or allergies.
- **Immune Dysfunction:** Overactive immune response leads to skin inflammation.
- **Skin Barrier Defect:** The skin fails to retain moisture, making it more vulnerable.
- **Environmental Triggers:** Dust, pollen, soaps, detergents, stress, weather changes.

### Symptoms
- **Itching (Pruritus):** Can be severe and worsen at night.
- **Dry, cracked, or scaly skin.**
- **Red or inflamed patches:** Common on **hands, neck, inner elbows, behind knees**.
- **Fluid-filled blisters that weep or crust over (in severe cases).**
- **Thickened, leathery skin (Lichenification)** from repeated scratching.

### Types of Eczema
1. Atopic Dermatitis:** Chronic, genetic form with flares and remissions.
2. Contact Dermatitis:** Triggered by irritants (*soaps, chemicals, metals*).
3. Dyshidrotic Eczema:** Small, **itchy blisters** on the hands and feet.
4. Nummular Eczema:** Coin-shaped, itchy lesions on the legs or arms.
5. Seborrheic Dermatitis:** Greasy, scaly patches on the scalp (dandruff).

### How to Manage and Treat Eczema?
#### 1. Skincare Routine
- Use **thick, fragrance-free moisturizers** (petroleum jelly, ceramide creams).
- Take **short lukewarm baths**, avoid long hot showers.
- Use **gentle, sulfate-free cleansers**.

#### 2. Medications
- **Topical steroids** (*Hydrocortisone, Clobetasol*): Reduce inflammation.
- **Calcineurin inhibitors** (*Tacrolimus, Pimecrolimus*): Used for sensitive areas.
- **Oral antihistamines** (*Diphenhydramine, Loratadine*): Help control itching.
- **Dupilumab (Dupixent®):** Biologic treatment for moderate-to-severe eczema.

#### 3. Lifestyle & Home Remedies
- **Wear breathable cotton clothing** (avoid wool, synthetic fabrics).
- **Avoid triggers** (dust, fragrances, stress).
- **Keep nails short** to prevent skin damage from scratching.

#### 4. Advanced Treatments
- **Phototherapy (UV Light Therapy)** for chronic eczema.
- **Biologic drugs targeting immune pathways**.
""",

    "Exanthems & Drug Eruptions": """**Exanthems & Drug Eruptions: Skin Reactions to Viruses and Medications**

### What are Exanthems?
Exanthems are widespread **skin rashes** caused by viral infections or **immune reactions to medications**. They often occur with fever and other systemic symptoms.

### Common Types of Exanthems
1. **Viral Exanthems (Infectious Rashes)**
   - **Measles (Rubeola):** Red, blotchy rash starting on the face and spreading down.
   - **Rubella (German Measles):** Mild pink rash with swollen lymph nodes.
   - **Roseola (Sixth Disease):** High fever followed by a **sudden pink rash**.
   - **Hand-Foot-Mouth Disease:** Small red blisters on hands, feet, and inside the mouth (caused by *Coxsackievirus*).
   - **Chickenpox (Varicella):** Itchy, fluid-filled blisters that crust over.

2. **Drug-Induced Exanthems**
   - A widespread rash caused by **medications**.
   - **Common triggers:** Antibiotics (*penicillins, sulfa drugs*), NSAIDs, anticonvulsants.
   - **Symptoms:** Red, **flat or raised rashes**, often itchy and symmetric.
   - Severe reactions include:
     - **Stevens-Johnson Syndrome (SJS) & Toxic Epidermal Necrolysis (TEN)** → Painful blisters, peeling skin, mucous membrane involvement (can be life-threatening).

### Symptoms of Exanthems
- Fever, sore throat, **malaise**.
- Rash **spreads symmetrically** across the body.
- Some cause **itching, peeling, or blistering**.

### Treatment & Management
- **For viral exanthems:**
  - Supportive care (*hydration, fever control*).
  - Antiviral medications (*Acyclovir for varicella*).
  - Avoid scratching to prevent scarring.

- **For drug eruptions:**
  - **Stop the offending drug** immediately.
  - **Antihistamines & corticosteroids** for symptom relief.
  - **Hospitalization for severe cases (SJS, TEN).**

### When to Seek Medical Attention?
- **High fever, difficulty breathing, severe pain.**
- **Blisters, peeling skin, or mucosal involvement.**
- **Rash that rapidly worsens or spreads.**
""",
"Hair Loss & Hair Diseases": """**Hair Loss & Hair Diseases: Causes, Symptoms, and Treatments**  

### Overview  
Hair diseases include conditions that affect **hair growth, scalp health, and hair structure**. Hair loss can be temporary or permanent and may result from **genetics, hormonal imbalances, autoimmune disorders, infections, or stress**.  

### Common Types of Hair Disorders  

#### 1. **Alopecia (Hair Loss)**  
Alopecia refers to **partial or complete hair loss**, and it can be caused by various factors:  
- **Androgenetic Alopecia (Male & Female Pattern Baldness):**  
  - **Genetic** condition leading to gradual hair thinning.  
  - **Men:** Receding hairline and crown thinning.  
  - **Women:** Widening hair part with diffuse thinning.  
  - **Treatment:** Minoxidil (Rogaine), Finasteride (for men), Hair Transplant.  

- **Alopecia Areata:**  
  - **Autoimmune disorder** where the immune system attacks hair follicles.  
  - Causes **round bald patches** on the scalp, beard, or eyebrows.  
  - Can progress to **Alopecia Totalis (complete scalp baldness)** or **Alopecia Universalis (full body hair loss)**.  
  - **Treatment:** Steroids, immunotherapy, JAK inhibitors.  

- **Telogen Effluvium:**  
  - **Stress-induced** hair shedding (illness, childbirth, medication).  
  - **Temporary condition**; hair regrows within 6 months.  
  - **Treatment:** Nutrition, stress management, biotin supplements.  

- **Traction Alopecia:**  
  - **Caused by tight hairstyles** (braids, ponytails, wigs).  
  - Leads to **receding hairline and thinning edges**.  
  - **Prevention:** Looser hairstyles, scalp massage.  

#### 2. **Scalp Infections & Hair Diseases**  

- **Tinea Capitis (Scalp Ringworm):**  
  - Fungal infection causing **scaly patches, hair breakage, and black dots**.  
  - Common in **children**.  
  - **Treatment:** Oral antifungals (*Griseofulvin, Terbinafine*).  

- **Seborrheic Dermatitis (Scalp Eczema):**  
  - **Dandruff, greasy scales, itching** (linked to Malassezia yeast).  
  - **Treatment:** Antifungal shampoos (*Ketoconazole, Zinc Pyrithione*).  

- **Folliculitis:**  
  - **Bacterial infection** of hair follicles, causing **red bumps or pustules**.  
  - **Treatment:** Antibiotics (*Mupirocin, Clindamycin*), warm compress.  

### Prevention & Management  
- **Use gentle hair products** (avoid sulfates, harsh chemicals).  
- **Reduce stress** (mindfulness, exercise).  
- **Eat a balanced diet** (protein, iron, vitamin D).  
- **Avoid excessive heat styling & tight hairstyles**.  
""",

    "Herpes, HPV & STDs": """**Herpes, HPV & Sexually Transmitted Diseases (STDs): Symptoms and Treatments**  

### Overview  
Sexually Transmitted Diseases (STDs) are infections **spread through sexual contact** (vaginal, oral, or anal sex). Some STDs can cause **long-term complications**, while others are curable with treatment.  

### 1. **Herpes (HSV-1 & HSV-2)**  
Herpes Simplex Virus (HSV) causes **painful blisters or sores** on the mouth or genitals.  
- **HSV-1:** Causes **cold sores** (oral herpes).  
- **HSV-2:** Causes **genital herpes** (sexually transmitted).  

**Symptoms:**  
- Painful **blisters** that turn into open sores.  
- Tingling or burning before an outbreak.  
- Flu-like symptoms (fever, swollen glands).  

**Treatment:**  
- **No cure, but antiviral medications** help control outbreaks (*Acyclovir, Valacyclovir*).  
- Avoid kissing or sex during outbreaks.  

### 2. **Human Papillomavirus (HPV)**  
HPV is the most common STD, causing **genital warts and cervical cancer**.  
- **Low-risk HPV:** Causes **warts** on the genitals, anus, or throat.  
- **High-risk HPV:** Can lead to **cervical, anal, or throat cancer**.  

**Prevention:**  
- **HPV vaccine (Gardasil 9)** protects against cancer-causing strains.  
- Regular **Pap smears** for early detection.  

**Treatment:**  
- Warts can be removed using **cryotherapy (freezing), laser, or topical treatments**.  

### 3. **Other Common STDs**  

- **Chlamydia & Gonorrhea:**  
  - **Symptoms:** Painful urination, abnormal discharge, pelvic pain.  
  - **Treatment:** Antibiotics (*Azithromycin, Doxycycline, Ceftriaxone*).  

- **Syphilis:**  
  - **Stages:**  
    1. **Primary:** Painless sore (chancre).  
    2. **Secondary:** Rash on palms and soles.  
    3. **Tertiary:** Can damage the brain, heart, and nerves.  
  - **Treatment:** Penicillin injection.  

- **HIV/AIDS:**  
  - Attacks the immune system, increasing infection risk.  
  - **Treatment:** Antiretroviral therapy (ART) for viral suppression.  

### Prevention & Safe Practices  
- **Use condoms** (reduces risk but not 100% protective).  
- **Regular STD testing** for sexually active individuals.  
- **HPV vaccination** for teens and adults.  
- **Antiviral or antibiotic treatment** as needed.  
""",

    "Light Disorders": """**Light Disorders: Skin Conditions Triggered by Light Exposure**  

### Overview  
Light disorders involve **abnormal skin reactions** to sunlight or artificial light sources. They may be caused by **genetic conditions, autoimmune responses, or medication sensitivity**.  

### 1. **Photosensitivity Reactions**  
Some people develop rashes, burns, or hives after **sun exposure**.  
- **Polymorphic Light Eruption (PLE):**  
  - **Itchy red bumps or blisters** hours after sun exposure.  
  - Common in **fair-skinned individuals**.  
  - **Treatment:** Sunscreens, antihistamines, phototherapy desensitization.  

- **Solar Urticaria:**  
  - **Hives and swelling** within minutes of sun exposure.  
  - **Treatment:** Antihistamines, sun avoidance, UV protection.  

### 2. **Photodermatitis (Sun Allergy)**  
- Caused by **certain medications, plants, or skin conditions**.  
- Symptoms include **red patches, swelling, peeling, and blisters**.  
- **Triggers:**  
  - **Medications:** Antibiotics (*Tetracyclines*), NSAIDs (*Ibuprofen*).  
  - **Plants:** Lime, celery, parsnip (phytophotodermatitis).  

**Treatment:**  
- **Stop the triggering drug or exposure.**  
- **Use broad-spectrum sunscreens (SPF 50+).**  
- **Corticosteroids for inflammation.**  

### 3. **Xeroderma Pigmentosum (XP) – Rare Genetic Disorder**  
- **Extreme sensitivity to UV light.**  
- High risk of **skin cancer and eye damage**.  
- **Lifelong UV protection** is needed.  

### Prevention & Management  
- **Avoid peak sun hours (10 AM - 4 PM).**  
- **Wear UV-protective clothing, sunglasses, and hats.**  
- **Use broad-spectrum sunscreen daily.**  
- **Seek shade whenever possible.**  

""",
 "Lupus & Connective Tissue Diseases": """**Lupus & Connective Tissue Diseases: Causes, Symptoms, and Treatments**  

### Overview  
Lupus and connective tissue diseases are **autoimmune disorders** that cause **inflammation, pain, and tissue damage** in the skin, joints, and internal organs. These conditions occur when the immune system mistakenly attacks healthy tissues.  

### 1. **Lupus (Systemic Lupus Erythematosus - SLE)**  
Lupus is a **chronic autoimmune disease** affecting the skin, joints, kidneys, heart, and brain.  

**Symptoms:**  
- **Butterfly-shaped rash** on the face.  
- **Fatigue, joint pain, swelling**.  
- **Photosensitivity** (skin rashes triggered by sunlight).  
- **Kidney problems (Lupus Nephritis)**.  
- **Mouth ulcers, chest pain, hair loss**.  

**Causes & Risk Factors:**  
- **Genetics, hormonal changes (estrogen), infections, and UV exposure**.  
- More common in **women (ages 15-45)**.  

**Treatment:**  
- **Anti-inflammatory medications (NSAIDs, corticosteroids).**  
- **Hydroxychloroquine (Plaquenil) for skin and joint symptoms.**  
- **Immunosuppressants (Methotrexate, Mycophenolate).**  

### 2. **Other Connective Tissue Diseases**  
- **Scleroderma:** Hardening and thickening of the skin, affects blood vessels.  
- **Dermatomyositis:** Muscle weakness with a skin rash (purple rash on eyelids, red knuckles).  
- **Sjogren’s Syndrome:** Dry eyes and mouth due to immune attack on glands.  

### Management & Prevention  
- **Avoid sun exposure, use sunscreen daily.**  
- **Regular checkups for kidney, heart, and lung health.**  
- **Balanced diet, exercise, and stress management.**  
""",

    "Melanoma & Moles": """**Melanoma & Moles: Identifying Skin Cancer Risk**  

### Overview  
Melanoma is the **deadliest form of skin cancer**, while **moles (nevi)** are common skin growths. Early detection is crucial for survival.  

### 1. **Moles (Nevi)**  
Moles are **clusters of pigmented cells** (melanocytes) that appear as dark spots on the skin.  
- **Common moles:** Brown, round, uniform in color.  
- **Atypical (Dysplastic) moles:** Larger, irregular, may indicate higher melanoma risk.  

### 2. **Melanoma: The Most Dangerous Skin Cancer**  
Melanoma develops when **melanocytes grow uncontrollably**.  

**Symptoms (ABCDE Rule):**  
- **A - Asymmetry:** One half of the mole doesn’t match the other.  
- **B - Border irregularity:** Uneven, jagged, or blurred edges.  
- **C - Color variation:** Multiple colors (brown, black, red, white).  
- **D - Diameter:** Larger than **6mm (pencil eraser size)**.  
- **E - Evolving:** Changes in size, shape, color, or bleeding.  

**Risk Factors:**  
- **Excessive sun exposure, UV radiation from tanning beds.**  
- **Fair skin, history of sunburns.**  
- **Family history of melanoma.**  

**Treatment:**  
- **Surgical removal (excision biopsy).**  
- **Immunotherapy (checkpoint inhibitors - Keytruda, Opdivo).**  
- **Targeted therapy (BRAF inhibitors for advanced cases).**  

### Prevention  
- **Wear sunscreen (SPF 50+), avoid tanning beds.**  
- **Monitor moles regularly, see a dermatologist for suspicious changes.**  
""",

    "Nail Fungus & Nail Diseases": """**Nail Fungus & Nail Diseases: Causes, Symptoms, and Treatments**  

### Overview  
Nail diseases affect the **growth, color, and structure of fingernails and toenails**. The most common condition is **nail fungus (onychomycosis)**.  

### 1. **Nail Fungus (Onychomycosis)**  
**Fungal infections** cause **thickened, discolored, and brittle nails**.  

**Symptoms:**  
- Yellow, white, or brown **nail discoloration**.  
- Thickened, crumbly, or brittle nails.  
- **Separation from the nail bed** (onycholysis).  

**Causes & Risk Factors:**  
- **Exposure to fungi (Tinea, Candida) in damp environments (showers, gyms).**  
- **Diabetes, poor circulation, weakened immune system.**  
- **Wearing tight shoes, artificial nails, or nail trauma.**  

**Treatment:**  
- **Oral antifungals (Terbinafine, Itraconazole).**  
- **Topical antifungals (Ciclopirox, Efinaconazole).**  
- **Laser therapy for persistent infections.**  

### 2. **Other Nail Diseases**  
- **Paronychia:** Bacterial infection causing **red, swollen skin around nails**.  
- **Psoriasis nails:** Pitting, ridges, and discoloration linked to psoriasis.  
- **Beau’s lines:** Horizontal ridges due to illness or stress.  

### Prevention  
- **Keep nails clean, dry, and trimmed.**  
- **Avoid walking barefoot in public showers.**  
- **Change socks frequently, disinfect nail clippers.**  
""",

    "Poison Ivy & Contact Dermatitis": """**Poison Ivy & Contact Dermatitis: Skin Reactions to Allergens**  

### Overview  
Contact dermatitis is a **skin reaction caused by irritants or allergens**, leading to **redness, itching, and rash formation**.  

### 1. **Poison Ivy, Oak, and Sumac**  
These plants contain **urushiol**, an **oily allergen** that causes **itchy, blistering rashes**.  

**Symptoms (24-48 hours after exposure):**  
- **Red, itchy rash with blisters** (spreads if scratched).  
- **Swelling, burning, and severe discomfort.**  
- **Fluid-filled blisters that break open and crust.**  

**Treatment:**  
- **Wash skin immediately** with soap and water to remove urushiol.  
- **Apply calamine lotion, antihistamines (Benadryl).**  
- **Severe cases:** Corticosteroids (Prednisone).  

### 2. **Other Types of Contact Dermatitis**  

- **Irritant Contact Dermatitis:**  
  - **Caused by soaps, detergents, cleaning chemicals.**  
  - **Symptoms:** Dry, cracked, burning skin.  
  - **Treatment:** Avoid irritants, use barrier creams.  

- **Allergic Contact Dermatitis:**  
  - **Triggered by nickel (jewelry), latex, cosmetics, or fragrances.**  
  - **Symptoms:** Red, swollen, itchy rash with possible blisters.  
  - **Treatment:** Identify allergen, apply steroids, take antihistamines.  

### Prevention  
- **Wear gloves when handling chemicals or poison ivy.**  
- **Wash clothes and skin after outdoor exposure.**  
- **Use hypoallergenic skincare products.**  

""",
"Psoriasis & Related Diseases": """**Psoriasis & Related Diseases: Chronic Skin Conditions and Their Management**  

### Overview  
Psoriasis is a **chronic autoimmune skin disease** that speeds up the skin cell cycle, leading to **red, scaly patches** on the skin. It is **not contagious** but has a strong genetic component. Related diseases include **psoriatic arthritis and pustular psoriasis**.  

### 1. **Types of Psoriasis**  
- **Plaque Psoriasis (Psoriasis Vulgaris):**  
  - Most common type (**80-90% of cases**).  
  - **Thick, red patches (plaques) covered with silvery-white scales**.  
  - Common on **scalp, elbows, knees, and lower back**.  

- **Guttate Psoriasis:**  
  - **Small, red, drop-like lesions** on the body.  
  - Often triggered by **strep throat infections**.  

- **Inverse Psoriasis:**  
  - **Red, shiny lesions in skin folds** (under breasts, armpits, groin).  
  - **Sweat and friction worsen symptoms**.  

- **Pustular Psoriasis:**  
  - **White pustules (blisters filled with pus) on red skin**.  
  - Can be **life-threatening (generalized pustular psoriasis).**  

- **Erythrodermic Psoriasis:**  
  - **Severe, widespread redness and peeling of the skin**.  
  - **Medical emergency** requiring immediate treatment.  

### 2. **Causes & Risk Factors**  
- **Autoimmune disorder:** The immune system mistakenly attacks skin cells.  
- **Triggers:** Stress, infections, skin injuries, smoking, cold weather.  
- **Genetics:** Runs in families; **40% of patients have a family history**.  

### 3. **Symptoms**  
- **Red, inflamed skin with thick, silvery scales**.  
- **Severe itching, burning, and pain**.  
- **Cracked skin that may bleed**.  
- **Nail psoriasis:** Thickened, pitted, or yellow nails.  
- **Psoriatic arthritis:** Joint pain, stiffness, and swelling.  

### 4. **Treatment Options**  
- **Topical Treatments:** Corticosteroids, vitamin D analogs (calcipotriol), coal tar.  
- **Phototherapy (UVB light):** Slows down skin cell production.  
- **Systemic Medications:** Methotrexate, Cyclosporine, Acitretin.  
- **Biologics (for severe cases):** Adalimumab (Humira), Secukinumab (Cosentyx).  

### 5. **Prevention & Lifestyle Changes**  
- **Moisturize daily to reduce scaling and dryness.**  
- **Avoid triggers (stress, infections, smoking, alcohol).**  
- **Sunlight exposure in moderation (natural vitamin D).**  

""",

    "Scabies & Infestations": """**Scabies & Infestations: Causes, Symptoms, and Treatment**  

### Overview  
Scabies is a **highly contagious skin infestation** caused by the **Sarcoptes scabiei mite**. The mite burrows under the skin, triggering **intense itching and rashes**. It spreads through **direct skin contact and contaminated clothing/bedding**.  

### 1. **Causes & Risk Factors**  
- **Caused by microscopic mites that lay eggs under the skin.**  
- **Spreads through prolonged skin-to-skin contact** (common in crowded living conditions).  
- **Can affect anyone, but more common in nursing homes, schools, and prisons.**  

### 2. **Symptoms (Develop 4-6 Weeks After Infection)**  
- **Severe itching (worse at night)**.  
- **Thin, wavy burrow tracks on the skin (gray or red lines).**  
- **Rash between fingers, wrists, elbows, waist, and genitals.**  
- **Blisters or crusted skin (in severe cases).**  

### 3. **Types of Scabies**  
- **Classic Scabies:** Affects small areas of the body with mild symptoms.  
- **Crusted (Norwegian) Scabies:** Severe form with thick, scaly skin and **thousands of mites** (highly contagious).  
- **Nodular Scabies:** Persistent itchy nodules, even after treatment.  

### 4. **Diagnosis**  
- **Skin scraping under a microscope to identify mites or eggs.**  
- **Dermoscopy to check for burrows.**  

### 5. **Treatment & Management**  
- **Topical medications:**  
  - **Permethrin 5% cream** (first-line treatment).  
  - **Lindane, Crotamiton, Sulfur ointment (alternative treatments).**  
- **Oral Medications:**  
  - **Ivermectin (for severe or resistant cases).**  
- **Antihistamines & Corticosteroids:** Reduce itching and inflammation.  

### 6. **Prevention & Hygiene**  
- **Wash all clothes, bedding, and towels in hot water.**  
- **Vacuum furniture and carpets thoroughly.**  
- **Treat all household members and close contacts to prevent reinfection.**  

""",

    "Seborrheic Keratoses & Benign Tumors": """**Seborrheic Keratoses & Benign Skin Tumors: Harmless Growths and How to Identify Them**  

### Overview  
Seborrheic keratoses are **non-cancerous (benign) skin growths** that appear as **brown, black, or tan wart-like patches**. They are common in **older adults** and often mistaken for melanoma.  

### 1. **Seborrheic Keratoses (SKs)**  
- **Appearance:**  
  - Round or oval, **rough or waxy surface**.  
  - **“Stuck-on” look** (raised above the skin).  
  - Can be **single or multiple lesions** on the face, chest, back, or scalp.  
- **Causes:**  
  - **Aging (most common in people over 50).**  
  - **Genetics (family history increases risk).**  
  - **Sun exposure may contribute but isn’t a direct cause.**  

**Treatment:**  
- **No medical treatment needed unless irritated or cosmetically unwanted.**  
- **Cryotherapy (freezing with liquid nitrogen).**  
- **Curettage (scraping off the lesion).**  
- **Laser therapy or electrosurgery for larger lesions.**  

### 2. **Other Benign Skin Tumors**  
- **Dermatofibroma:** Firm, small nodules on the skin caused by minor injuries or insect bites.  
- **Lipomas:** Soft, fatty lumps under the skin (harmless but can be removed if painful).  
- **Cherry Angiomas:** Small, bright red mole-like spots made of blood vessels.  

### 3. **How to Differentiate Benign Tumors from Skin Cancer**  
- **Seborrheic Keratoses vs. Melanoma:**  
  - **SKs are waxy and stuck-on, while melanoma is irregular in shape and grows rapidly.**  
- **If a lesion changes in color, shape, or size, a biopsy is recommended.**  

### 4. **Prevention & Monitoring**  
- **Avoid excessive sun exposure (use sunscreen).**  
- **Regular skin checks to identify new growths.**  
- **See a dermatologist for any suspicious or rapidly changing lesions.**  

""",
"Systemic Disease": """**Systemic Diseases and Their Impact on the Skin**  

### Overview  
Systemic diseases are **medical conditions that affect multiple organs or the entire body**, often with **skin-related symptoms**. These diseases include **autoimmune disorders, metabolic conditions, and infections** that manifest through **rashes, discoloration, ulcers, or other skin changes**.  

### 1. **Common Systemic Diseases with Skin Manifestations**  

- **Diabetes Mellitus:**  
  - **Skin symptoms:** Darkened skin patches (**acanthosis nigricans**), slow-healing wounds, bacterial & fungal infections.  
  - **Cause:** High blood sugar damages blood vessels and weakens the immune system.  

- **Lupus (Systemic Lupus Erythematosus - SLE):**  
  - **Skin symptoms:** Butterfly-shaped rash on the face, hair loss, ulcers in the mouth.  
  - **Cause:** The immune system mistakenly attacks healthy tissue.  

- **Scleroderma:**  
  - **Skin symptoms:** Thickened, tight, shiny skin, Raynaud’s phenomenon (cold-sensitive fingers turning blue).  
  - **Cause:** Overproduction of collagen, leading to skin hardening.  

- **Rheumatoid Arthritis (RA):**  
  - **Skin symptoms:** Nodules under the skin, ulcers, red patches due to inflammation of blood vessels (**vasculitis**).  

- **Kidney Disease:**  
  - **Skin symptoms:** Itchy, dry skin due to toxin buildup, **yellowish or gray skin tone**.  

### 2. **Symptoms of Systemic Diseases Affecting the Skin**  
- **Unexplained skin rashes or discoloration.**  
- **Itchy, dry, or scaling skin that doesn’t respond to normal treatments.**  
- **Sudden skin changes (e.g., ulcers, swelling, or hair loss) associated with fever or joint pain.**  
- **Dark patches, yellowing, or purplish spots indicating internal disease.**  

### 3. **Diagnosis**  
- **Blood tests** to check immune function, blood sugar, and organ health.  
- **Skin biopsy** to examine unusual rashes or growths.  
- **Imaging tests (X-ray, MRI)** for deeper organ involvement.  

### 4. **Treatment & Management**  
- **Treat the underlying disease** (e.g., insulin for diabetes, immunosuppressants for lupus).  
- **Manage skin symptoms:** Moisturizers for dryness, steroid creams for inflammation.  
- **Lifestyle changes:** Healthy diet, stress reduction, and regular exercise.  

### 5. **Prevention & Monitoring**  
- **Early detection and management of systemic conditions prevent severe complications.**  
- **Routine medical check-ups for people with chronic diseases.**  
- **Use sunscreen to protect sensitive skin from UV rays, especially in autoimmune diseases.**  

""",

    "Fungal Infections": """**Fungal Infections: Causes, Symptoms, and Treatment**  

### Overview  
Fungal infections are **caused by fungi (molds, yeasts, and dermatophytes)** that grow on the skin, nails, or mucous membranes. They thrive in **warm, moist areas like skin folds, feet, and the groin**.  

### 1. **Common Types of Fungal Infections**  

- **Tinea (Ringworm):**  
  - **Circular, red, scaly patches with a raised border.**  
  - Affects scalp (**tinea capitis**), body (**tinea corporis**), feet (**athlete’s foot/tinea pedis**), and groin (**jock itch/tinea cruris**).  

- **Candidiasis (Yeast Infection):**  
  - **White patches inside the mouth (oral thrush)** or **red, itchy skin folds**.  
  - Common in **diabetics, immunocompromised patients, and those taking antibiotics**.  

- **Onychomycosis (Nail Fungus):**  
  - **Thick, discolored, brittle nails.**  
  - **Slow-growing and difficult to treat.**  

### 2. **Causes & Risk Factors**  
- **Damp, humid environments (sweaty feet, tight shoes).**  
- **Weakened immune system (HIV, cancer, diabetes).**  
- **Skin contact with infected people, animals, or surfaces (gym floors, showers).**  

### 3. **Symptoms**  
- **Red, itchy, scaly rashes.**  
- **Peeling, cracking, or blistering skin.**  
- **Foul-smelling, thickened nails in fungal nail infections.**  

### 4. **Diagnosis**  
- **Microscopic examination of skin scrapings.**  
- **Fungal culture to identify the specific fungus.**  

### 5. **Treatment Options**  
- **Topical antifungal creams:** Clotrimazole, Terbinafine, Miconazole.  
- **Oral antifungal pills (for severe cases):** Fluconazole, Itraconazole.  
- **Keeping the affected area dry and clean to prevent recurrence.**  

### 6. **Prevention & Hygiene Tips**  
- **Wear breathable shoes and change socks daily.**  
- **Avoid sharing personal items (towels, socks, nail clippers).**  
- **Use antifungal powders in shoes if prone to athlete’s foot.**  

""",

    "Urticaria (Hives)": """**Urticaria (Hives): Causes, Symptoms, and Management**  

### Overview  
Urticaria (hives) is a **sudden outbreak of itchy, red, swollen welts** on the skin caused by an allergic reaction or other triggers. It is **not contagious** but can be **chronic (lasting more than 6 weeks)**.  

### 1. **Types of Urticaria**  
- **Acute Urticaria:** Lasts less than 6 weeks, often due to food allergies, medications, or infections.  
- **Chronic Urticaria:** Persistent hives with **no clear allergic trigger**; linked to autoimmune diseases.  
- **Physical Urticaria:** Triggered by **cold, heat, exercise, sunlight, or pressure on the skin**.  

### 2. **Causes & Triggers**  
- **Allergic reactions (foods, drugs, insect bites, pollen).**  
- **Physical triggers (heat, cold, sun exposure, exercise).**  
- **Infections (viral, bacterial, or parasitic infections).**  
- **Emotional stress or autoimmune diseases (e.g., thyroid disorders).**  

### 3. **Symptoms**  
- **Raised, red, itchy welts that may change shape and location.**  
- **Swelling of lips, eyelids, or face (angioedema in severe cases).**  
- **Burning or stinging sensation in affected areas.**  

### 4. **Diagnosis**  
- **Skin allergy tests** to identify triggers.  
- **Blood tests for autoimmune markers in chronic urticaria.**  

### 5. **Treatment & Management**  
- **Antihistamines (Cetirizine, Loratadine, Diphenhydramine) to relieve itching.**  
- **Corticosteroids for severe reactions.**  
- **Epinephrine injection (EpiPen) in case of anaphylaxis (life-threatening allergic reaction).**  

### 6. **Prevention & Lifestyle Changes**  
- **Identify and avoid known allergens or triggers.**  
- **Keep skin cool and avoid tight clothing that can cause pressure urticaria.**  
- **Reduce stress, as emotional triggers can worsen chronic hives.**  

""",
"Vascular Tumors": """**Vascular Tumors: Types, Causes, Symptoms, and Treatment**  

### Overview  
Vascular tumors are **abnormal growths of blood vessels or lymphatic vessels** that can be **benign (non-cancerous) or malignant (cancerous)**. These tumors **form due to excessive blood vessel proliferation** and can appear **on the skin, internal organs, or deeper tissues**.  

### 1. **Types of Vascular Tumors**  

#### **Benign Vascular Tumors (Non-Cancerous)**  
- **Hemangiomas:**  
  - **Appearance:** Red, blue, or purple birthmark-like growths.  
  - **Common in infants (infantile hemangiomas).**  
  - Usually **shrink on their own** but may require treatment if they obstruct vision or breathing.  

- **Pyogenic Granulomas:**  
  - **Small, bright red growths that bleed easily.**  
  - **Triggered by trauma or hormonal changes (e.g., pregnancy).**  

#### **Malignant Vascular Tumors (Cancerous or Aggressive Growths)**  
- **Angiosarcoma:**  
  - **Aggressive cancer that originates in blood vessels.**  
  - Appears as **bruised or purplish skin lesions** that grow and ulcerate.  
  - **Requires immediate treatment (surgery, chemotherapy).**  

- **Kaposi Sarcoma:**  
  - **Caused by human herpesvirus 8 (HHV-8).**  
  - Common in **HIV/AIDS patients** and presents as **purple or dark red nodules**.  
  - Treated with **antiviral medications, radiation, or chemotherapy**.  

### 2. **Causes & Risk Factors**  
- **Genetic mutations affecting blood vessel growth.**  
- **Hormonal changes (pregnancy can trigger some vascular tumors).**  
- **Radiation exposure (linked to angiosarcoma).**  
- **Immune system suppression (e.g., HIV/AIDS leading to Kaposi Sarcoma).**  

### 3. **Symptoms**  
- **Red, blue, or purple skin lesions that grow over time.**  
- **Bleeding or ulceration in certain tumors.**  
- **Swelling and pain if the tumor compresses surrounding tissues.**  

### 4. **Diagnosis**  
- **Physical examination & dermoscopy** to assess tumor structure.  
- **Biopsy** to confirm if it is benign or malignant.  
- **Imaging tests (MRI, CT scans)** for deeper or internal tumors.  

### 5. **Treatment & Management**  
- **Observation:** Small hemangiomas in infants often shrink without treatment.  
- **Medications:** Beta-blockers like propranolol can reduce hemangiomas.  
- **Laser therapy:** Used to shrink superficial vascular tumors.  
- **Surgery or chemotherapy:** Required for malignant vascular tumors like angiosarcoma.  

""",

    "Vasculitis": """**Vasculitis: Causes, Symptoms, and Treatment**  

### Overview  
Vasculitis is **inflammation of blood vessels**, leading to **narrowing, weakening, or blockage of arteries and veins**. This can **reduce blood flow to organs**, causing **skin rashes, nerve problems, and organ damage**.  

### 1. **Types of Vasculitis**  
- **Small-Vessel Vasculitis:**  
  - Affects tiny blood vessels in **skin, kidneys, and lungs**.  
  - Includes **Henoch-Schönlein Purpura (HSP)** and **ANCA-associated vasculitis**.  

- **Medium-Vessel Vasculitis:**  
  - Includes **Polyarteritis Nodosa (PAN)**, which affects the **skin, joints, nerves, and digestive system**.  

- **Large-Vessel Vasculitis:**  
  - Includes **Giant Cell Arteritis (GCA)**, which affects **head arteries, causing headaches and vision problems**.  

### 2. **Causes & Risk Factors**  
- **Autoimmune diseases (Lupus, Rheumatoid Arthritis).**  
- **Infections (Hepatitis B & C can trigger vasculitis).**  
- **Certain medications (antibiotics, NSAIDs).**  

### 3. **Symptoms**  
- **Skin rash (red or purple spots due to bleeding under the skin).**  
- **Fever, fatigue, weight loss.**  
- **Joint pain, muscle weakness.**  
- **Nerve issues (numbness, tingling, or pain).**  

### 4. **Diagnosis**  
- **Blood tests for inflammation markers (ESR, CRP).**  
- **Skin biopsy to check for blood vessel inflammation.**  
- **Angiography (X-ray of blood vessels) to detect narrowing or blockage.**  

### 5. **Treatment & Management**  
- **Corticosteroids (Prednisone) to reduce inflammation.**  
- **Immunosuppressants (Methotrexate, Azathioprine) for severe cases.**  
- **Plasma exchange therapy (in severe autoimmune-related vasculitis).**  

""",
    "Warts & Viral Infections": """**Warts & Viral Infections: Causes, Symptoms, and Treatment**  

### Overview  
Warts are **non-cancerous skin growths caused by the Human Papillomavirus (HPV)**. They appear as **rough, raised bumps** on the skin and can be **contagious** through direct or indirect contact.  

### 1. **Types of Warts**  
- **Common Warts (Verruca Vulgaris):**  
  - Appear on **fingers, hands, and knees**.  
  - Rough, **flesh-colored or grayish bumps**.  

- **Plantar Warts:**  
  - Found on the **soles of feet**.  
  - Painful due to inward growth from walking pressure.  

- **Flat Warts (Verruca Plana):**  
  - Small, smooth, **flesh-colored warts**, usually on the **face, neck, and hands**.  

- **Genital Warts:**  
  - Appear in the **genital area** due to **HPV strains 6 & 11**.  
  - Transmitted through **sexual contact**.  

### 2. **Causes & Risk Factors**  
- **HPV infection through skin contact or shared surfaces (showers, towels, razors).**  
- **Weakened immune system increases wart susceptibility.**  
- **Nail biting or skin picking can spread warts.**  

### 3. **Symptoms**  
- **Raised, rough skin growths that may itch or cause discomfort.**  
- **Plantar warts cause pain when walking.**  
- **Genital warts appear as small, flesh-colored bumps in clusters.**  

### 4. **Diagnosis**  
- **Physical examination by a dermatologist.**  
- **Dermoscopy to differentiate warts from other skin conditions.**  
- **HPV testing in cases of genital warts.**  

### 5. **Treatment & Management**  
- **Cryotherapy (Freezing with liquid nitrogen) to remove warts.**  
- **Salicylic acid treatments for common warts.**  
- **Laser therapy for stubborn or deep-seated warts.**  
- **Prescription creams (Imiquimod, Podophyllotoxin) for genital warts.**  

### 6. **Prevention & Hygiene Tips**  
- **Avoid direct contact with warts or contaminated surfaces.**  
- **Wear sandals in public showers to prevent plantar warts.**  
- **HPV vaccination (Gardasil) to prevent genital warts.**  

"""    
}

if len(CLASS_LABELS) != 23:
    print(f"⚠ Warning: Expected 23 class labels, but found {len(CLASS_LABELS)}.")

# Load MobileNetV2 as a feature extractor for skin check
feature_extractor = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# Function to check if the uploaded file is allowed
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to detect if the image contains skin
def is_skin_image(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Error: Image not found at {img_path}")
            return False

        img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(img_YCrCb, lower, upper)
        skin_pixels = cv2.countNonZero(mask)
        total_pixels = img.shape[0] * img.shape[1]
        skin_ratio = skin_pixels / total_pixels

        return skin_ratio > 0.20  # Adjusted threshold
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return False

# Function to preprocess an image for classification
def preprocess_image(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f" Error: Image not found at {img_path}")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f" Error processing image: {e}")
        return None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predicts")
def predicts():
    return render_template("predict.html")

@app.route("/about_us")
def about_us():
    return render_template("about.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")
@app.route("/dashboard")
def dashboard():
    if "user_name" in session:
        user_name = session["user_name"]  # Get user's name from session
    else:
        user_name = "Guest"  # Default for guests

    return render_template("dashboard.html", username=user_name)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route('/static/uploads/<filename>')
def uploaded_files(filename):
    return send_from_directory(app.config["UPLOAD_FOLDERS"], filename)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Check if the image contains enough skin before classification
        if not is_skin_image(file_path):
            return jsonify({"error": "No significant skin detected. Please upload a valid skin image."}), 400

        image_url = f"/uploads/{filename}"
        img_array = preprocess_image(file_path)
        if img_array is None:
            return jsonify({"error": "Invalid image format"}), 400

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        confidence = np.max(predictions) * 100  # Convert to percentage
        predicted_class = CLASS_LABELS[predicted_class_index]

        # Get disease description
        disease_description = DISEASE_DESCRIPTIONS.get(predicted_class, "Description not available.")

        # If confidence is too low (e.g., below 50%), assume healthy skin
        if confidence < 15:
            return jsonify({
                "disease": "No skin condition detected",
                "confidence": f"{confidence:.2f}%",
                "description": "You are okay! No significant skin disease was detected in the image.",
                "image_url": image_url
            })

        result = {
            "disease": predicted_class,
            "confidence": f"{confidence:.2f}%",
            "description": disease_description,
            "image_url": image_url
        }

        return jsonify(result)
    

    # MySQL Configuration (Update with your database details)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Kenya@1000'
app.config['MYSQL_DB'] = 'users'
app.config['SECRET_KEY'] = 'temporary_secret'

mysql = MySQL(app)
bcrypt = Bcrypt(app)

@app.route('/Uploadimages', methods=['POST'])
def Uploadimages():
    print("📌 Route '/Uploadimages' triggered")

    if 'user_id' not in session:
        print("❌ User not authenticated")
        return jsonify({"error": "User not authenticated"}), 401

    user_id = session['user_id']
    print(f"✅ User ID: {user_id}")

    if 'image' not in request.files:
        print("❌ No file uploaded")
        return jsonify({"error": "No file uploaded"}), 400

    image = request.files['image']
    if image.filename == '':
        print("❌ No selected file")
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(image.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDERS'], filename)
    image.save(filepath)
    print(f"✅ File saved: {filename} -> {filepath}")

    try:
        print("📌 Entering try block")
        cursor = mysql.connection.cursor()

        # Check MySQL connection
        cursor.execute("SELECT 1")
        print("✅ MySQL Connection is working")

        # Insert image details into database
        cursor.execute(
            "INSERT INTO user_images (user_id, image_filename, image_path, uploaded_at) VALUES (%s, %s, %s, NOW())",
            (user_id, filename, filepath)
        )

        mysql.connection.commit()
        cursor.close()
        print("✅ Image inserted into database")

    except Exception as e:
        print("❌ MySQL Connection Error:", str(e))
        return jsonify({"success": False, "message": str(e)}), 500

    # **Skin Classification Logic**
    print("📌 Checking if image contains enough skin...")
    if not is_skin_image(filepath):
        return jsonify({"error": "No significant skin detected. Please upload a valid skin image."}), 400

    print("📌 Preprocessing image for classification...")
    img_array = preprocess_image(filepath)
    if img_array is None:
        return jsonify({"error": "Invalid image format"}), 400

    print("📌 Running model prediction...")
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    predicted_class = CLASS_LABELS[predicted_class_index]

    disease_description = DISEASE_DESCRIPTIONS.get(predicted_class, "Description not available.")

    # Confidence threshold for healthy skin
    if confidence < 15:
        return jsonify({
            "disease": "No skin condition detected",
            "confidence": f"{confidence:.2f}%",
            "description": "You are okay! No significant skin disease was detected in the image.",
            "image_url": f"/static/uploads/{filename}"
        })

    # **Insert prediction into database**
    try:
        print("📌 Storing prediction in database...")
        cursor = mysql.connection.cursor()

        cursor.execute(
            "INSERT INTO user_predictions (user_id, image_filename, predicted_disease, confidence) VALUES (%s, %s, %s, %s)",
            (user_id, filename, predicted_class, confidence)
        )

        mysql.connection.commit()
        cursor.close()
        print("✅ Prediction stored successfully!")

    except Exception as e:
        print("❌ Error inserting prediction into database:", str(e))

    result = {
        "disease": predicted_class,
        "confidence": f"{confidence:.2f}%",
        "description": disease_description,
        "image_url": f"/static/uploads/{filename}"
    }

    return jsonify(result)

    

# Register Route
@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.json  # Get JSON data from frontend
        name = data['name']
        id_number = data['id_number']
        address = data['address']
        phone = data['phone']
        age = data['age']
        gender = data['gender']
        username = data['username']
        password = bcrypt.generate_password_hash(data['password']).decode('utf-8')

        cursor = mysql.connection.cursor()
        # Check if username or id_number already exists
        cursor.execute("SELECT username, id_number FROM register WHERE username = %s OR id_number = %s", (username, id_number))
        existing_user = cursor.fetchone()


        if existing_user:
            if existing_user[0] == username:
                return jsonify({"success": False, "message": "Username already exists. Try another one."}), 400
            if existing_user[1] == id_number:
                return jsonify({"success": False, "message": "ID Number already exists. Use a different one."}), 400

        # Insert user into database
        cursor.execute("INSERT INTO register (name, id_number, address, phone, age, gender, username, password) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", 
                       (name, id_number, address, phone, age, gender, username, password))
        mysql.connection.commit()
        cursor.close()

        return jsonify({"success": True, "message": "Registration Successful! proceed to Login now...."})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# Login Route
@app.route('/logins', methods=['POST'])
def logins():
    try:
        data = request.json  # Get JSON data from frontend
        username = data['username']
        password = data['password']

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT * FROM register WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()

        if user and bcrypt.check_password_hash(user["password"], password):
            # Store user info in session
            session["user_id"] = user["id"]  # Assuming 'id' is the primary key
            session["user_name"] = user["name"]  # Storing the full name
            return jsonify({"success": True, "message": "Login successful!", "username": user["username"]})
        else:
            return jsonify({"success": False, "message": "Invalid username or password."}), 401

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/fetch_records', methods=['GET'])
def fetch_records():
    if 'user_id' not in session:
        return jsonify({"error": "User not authenticated"}), 401

    user_id = session['user_id']
    
    try:
        cursor = mysql.connection.cursor()
        
        # Fetch user diagnosis records from the database
        cursor.execute("""
            SELECT image_filename, predicted_disease, confidence 
            FROM user_predictions 
            WHERE user_id = %s
            ORDER BY predicted_at DESC
        """, (user_id,))
        records = cursor.fetchall()

        # Close the cursor
        cursor.close()

        # Prepare records in a format that can be used in the template
        records_list = [{
            'image_url': f"/static/uploads/{record[0]}",  # Assuming image_filename is stored in the user_predictions table
            'disease': record[1],
            'confidence': record[2]
        } for record in records]

        return jsonify({'records': records_list})

    except Exception as e:
        print(f"❌ Error fetching records: {str(e)}")
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)
