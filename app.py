from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json
from deep_translator import GoogleTranslator

app = Flask(__name__)
CORS(app)

print("Loading model...")
model = load_model('model/best_model.h5')
print("Model loaded!")

CLASS_NAMES = [
    'Apple - Apple Scab', 'Apple - Black Rot', 'Apple - Cedar Apple Rust', 'Apple - Healthy',
    'Blueberry - Healthy', 'Cherry - Powdery Mildew', 'Cherry - Healthy',
    'Corn - Cercospora Leaf Spot', 'Corn - Common Rust', 'Corn - Northern Leaf Blight', 'Corn - Healthy',
    'Grape - Black Rot', 'Grape - Esca', 'Grape - Leaf Blight', 'Grape - Healthy',
    'Orange - Citrus Greening', 'Peach - Bacterial Spot', 'Peach - Healthy',
    'Pepper - Bacterial Spot', 'Pepper - Healthy',
    'Potato - Early Blight', 'Potato - Late Blight', 'Potato - Healthy',
    'Raspberry - Healthy', 'Soybean - Healthy', 'Squash - Powdery Mildew',
    'Strawberry - Leaf Scorch', 'Strawberry - Healthy',
    'Tomato - Bacterial Spot', 'Tomato - Early Blight', 'Tomato - Late Blight',
    'Tomato - Leaf Mold', 'Tomato - Septoria Leaf Spot',
    'Tomato - Spider Mites', 'Tomato - Target Spot',
    'Tomato - Yellow Leaf Curl Virus', 'Tomato - Mosaic Virus', 'Tomato - Healthy',
    'Not a Leaf'
]

# ── DISEASE DATABASE (English) ────────────────────────────────────────────────
DISEASE_DB = {
    'Apple - Apple Scab': {
        'severity': 'medium',
        'symptoms': ['Olive-green to brown velvety lesions on leaves','Dark scabby patches on fruit surface','Premature leaf drop in heavy infections','Distorted and cracked fruit with corky texture','Pale green patches visible on young lesions initially'],
        'treatment': ['Apply fungicide at bud break before green tissue appears','Remove and destroy all fallen leaves to eliminate overwintering spores','Prune canopy during dormant season to improve airflow','Use myclobutanil or captan sprays every 7 days during wet periods'],
        'prevention': ['Rake and destroy fallen leaves every autumn','Prune trees annually for open canopy and good airflow','Choose scab-resistant apple varieties when replanting','Avoid wetting foliage with overhead irrigation']
    },
    'Apple - Black Rot': {
        'severity': 'high',
        'symptoms': ['Brown rotting lesions on fruit turning completely black','Frog-eye spots on leaves with purple borders and tan centers','Cankers forming on limbs and branches','Mummified fruit hanging on tree through winter','Premature defoliation in severe cases'],
        'treatment': ['Remove all mummified fruit immediately as they are the primary spore source','Prune out all dead or cankered limbs and destroy off-site','Apply fungicide from pink bud stage through petal fall','Maintain tree vigor with proper fertilization'],
        'prevention': ['Remove mummified fruit before spring each year','Prune dead wood annually during dormant season','Choose resistant varieties where possible','Keep tools clean and disinfected between trees']
    },
    'Apple - Cedar Apple Rust': {
        'severity': 'medium',
        'symptoms': ['Bright orange-yellow spots on upper leaf surface','Pale yellow lesions with orange borders on leaves','Tube-like structures visible on leaf undersides','Similar lesions developing on fruit','Requires both cedar and apple trees to complete life cycle'],
        'treatment': ['Apply fungicide from pink bud stage through first cover spray','Remove nearby cedar and juniper trees if feasible','Time fungicide sprays before expected rain periods','Use myclobutanil or trifloxystrobin-based products'],
        'prevention': ['Plant rust-resistant apple varieties','Remove cedar and juniper trees near the orchard','Scout cedar trees for orange galls each spring','Apply preventive fungicide before wet weather']
    },
    'Apple - Healthy': {
        'severity': 'low',
        'symptoms': ['Glossy deep green leaves with no spots or lesions','No signs of cankers mummies or powdery coating','Healthy flower and fruit development progressing normally','Vigorous shoot growth with normal leaf size','No pest damage webbing or discoloration visible'],
        'treatment': ['Continue your current pruning and spray program','Monitor weekly for any early signs of disease','Maintain balanced fertilization throughout the season','Thin fruit if overcrowded to improve size and airflow'],
        'prevention': ['Maintain annual dormant pruning for open canopy','Continue preventive fungicide schedule each season','Rake fallen leaves every autumn without fail','Keep irrigation off foliage to reduce disease risk']
    },
    'Blueberry - Healthy': {
        'severity': 'low',
        'symptoms': ['Vibrant green leaves with no spots or lesions','Healthy berry clusters developing normally','Strong cane growth with normal leaf size','No signs of mummified berries or stem dieback','No pest damage or webbing present'],
        'treatment': ['Continue regular watering and fertilization','Monitor for early signs of mummy berry or botrytis','Prune out any weak or crossing canes','Maintain mulch layer to retain moisture'],
        'prevention': ['Maintain soil pH between 4.5 and 5.5 for optimal health','Apply mulch to suppress weeds and retain moisture','Prune annually to maintain open bush structure','Scout weekly during fruiting season']
    },
    'Cherry - Powdery Mildew': {
        'severity': 'medium',
        'symptoms': ['White powdery coating on young leaves and shoot tips','Leaves curl upward and become distorted','Infected leaves turn yellow then drop prematurely','Young fruit may develop russet patches','Shoot tips stunted and deformed in severe cases'],
        'treatment': ['Apply sulfur or potassium bicarbonate at first sign','Remove and destroy all infected tissue immediately','Apply myclobutanil or trifloxystrobin for established infections','Improve canopy airflow by selective pruning'],
        'prevention': ['Prune annually for good canopy airflow','Begin preventive sulfur sprays early in the season','Choose powdery mildew resistant varieties','Avoid excess nitrogen fertilization which promotes lush growth']
    },
    'Cherry - Healthy': {
        'severity': 'low',
        'symptoms': ['Glossy deep green leaves with no powdery coating','No spots lesions or shot-holes on leaves','Healthy flower and fruit development','Normal vigorous shoot extension each season','No signs of gummosis or cankers on branches'],
        'treatment': ['Continue annual dormant pruning program','Maintain preventive fungicide spray schedule','Monitor calcium levels for fruit quality','Thin fruit if overcrowded for better size'],
        'prevention': ['Prune annually for open vase canopy structure','Protect fruit from birds with netting','Maintain consistent irrigation throughout season','Scout weekly during key growth stages']
    },
    'Corn - Cercospora Leaf Spot': {
        'severity': 'medium',
        'symptoms': ['Small rectangular gray to tan lesions with dark borders','Lesions run parallel to leaf veins giving a banded appearance','Severe infections cause large areas of leaf tissue to die','Lower leaves affected first then progressing upward','Yield loss occurs when upper leaves are heavily infected'],
        'treatment': ['Apply strobilurin or triazole fungicide at VT stage','Scout from V6 growth stage onwards for early detection','Manage crop residue by tilling after harvest','Reduce plant stress with adequate nitrogen and water'],
        'prevention': ['Plant resistant hybrid varieties as primary defense','Rotate crops away from corn for at least one season','Manage residue to reduce overwintering spore load','Avoid dense planting that reduces airflow']
    },
    'Corn - Common Rust': {
        'severity': 'medium',
        'symptoms': ['Small cinnamon-brown to brick-red powdery pustules on both leaf surfaces','Pustules are elongated and rupture to release spores','Severe infection causes leaf yellowing and premature death','Husks and stalks can also be infected in severe cases','Spores spread rapidly by wind across large areas'],
        'treatment': ['Apply triazole fungicide at first pustule appearance','Scout crops during silking stage when risk is highest','Apply preventive fungicide at VT or R1 growth stage','Reduce plant stress with adequate nitrogen and irrigation'],
        'prevention': ['Plant rust-resistant hybrid varieties','Plant early to avoid peak spore pressure periods','Maintain good overall crop health and nutrition','Monitor disease forecasts and scout fields regularly']
    },
    'Corn - Northern Leaf Blight': {
        'severity': 'medium',
        'symptoms': ['Long tan to grayish-green elliptical lesions up to 15 cm','Dark sporulation visible in the center of mature lesions','Upper leaves infected first with disease progressing downward','Severe cases can cause significant yield reduction','Lesions may coalesce causing large dead areas on leaves'],
        'treatment': ['Apply propiconazole or azoxystrobin fungicide at VT stage','Choose resistant hybrid varieties for future plantings','Till infected crop residue into soil after harvest','Scout fields from V6 stage onwards for early detection'],
        'prevention': ['Select resistant hybrid varieties as the main strategy','Rotate crops with non-host species for one season','Manage crop residue by tillage after harvest','Plant at optimal density to maintain good airflow']
    },
    'Corn - Healthy': {
        'severity': 'low',
        'symptoms': ['Uniform deep green leaf color throughout the canopy','No lesions spots or discoloration anywhere','Healthy tassels and ear development progressing normally','Strong upright plant structure with good standability','No signs of pest damage or pathogen activity'],
        'treatment': ['Continue balanced fertilization especially nitrogen','Maintain consistent soil moisture throughout the season','Scout weekly for any early signs of disease or pests','Manage weeds to reduce competition and disease reservoirs'],
        'prevention': ['Maintain consistent irrigation throughout growth stages','Apply balanced NPK fertilization program','Rotate crops annually to break disease cycles','Plant at optimal density for your hybrid and environment']
    },
    'Grape - Black Rot': {
        'severity': 'high',
        'symptoms': ['Tan circular spots with dark brown borders on leaves','Fruit turns brown then wrinkles into hard black mummies','Lesions also appear on shoots and tendrils','White pycnidia visible as tiny dots inside leaf spots','Rapid and complete fruit rot during warm wet seasons'],
        'treatment': ['Apply mancozeb or myclobutanil from bud break through fruit set','Remove and destroy all mummified berries before spring','Prune out infected canes during dormant season','Remove leaves around fruit clusters to improve spray coverage'],
        'prevention': ['Remove all mummified fruit during winter pruning','Prune for open canopy to improve airflow and spray coverage','Begin protective sprays before bloom stage','Scout vineyards regularly especially during wet periods']
    },
    'Grape - Esca': {
        'severity': 'high',
        'symptoms': ['Tiger stripe pattern of yellow and red between leaf veins','Berries develop dark spots known as black measles','Sudden vine collapse can occur during summer heat','Internal wood shows brown to black streaking when cut','Progressive trunk disease that worsens over many years'],
        'treatment': ['No effective chemical cure exists for established infection','Protect all pruning wounds with wound sealant immediately','Remove severely affected vines to prevent spread to neighbors','Delay all pruning until dry weather periods only'],
        'prevention': ['Prune only during dry weather to prevent spore infection','Seal all pruning wounds immediately after cutting','Use certified disease-free planting material only','Monitor vines for early tiger stripe symptoms each season']
    },
    'Grape - Leaf Blight': {
        'severity': 'medium',
        'symptoms': ['Angular dark brown spots on leaves with yellow halos','Spots coalesce causing large necrotic areas on leaf','Premature defoliation weakens vines significantly','Lesions have dark fungal sporulation visible on surface','Disease progresses rapidly in warm humid conditions'],
        'treatment': ['Apply copper-based fungicide as preventive and curative treatment','Remove infected leaves to reduce spore load in the vineyard','Improve canopy airflow with leaf removal and shoot positioning','Avoid all overhead irrigation to keep foliage dry'],
        'prevention': ['Maintain open canopy through regular pruning','Use drip irrigation to keep foliage dry at all times','Scout vineyards regularly throughout the growing season','Remove fallen leaves to reduce overwintering inoculum']
    },
    'Grape - Healthy': {
        'severity': 'low',
        'symptoms': ['Uniform green leaf color with no spots or lesions','Healthy fruit cluster development progressing normally','Vigorous shoot growth with normal internode length','No signs of trunk disease gummosis or dieback','No pest damage or powdery coating anywhere'],
        'treatment': ['Continue regular shoot positioning and leaf removal','Maintain balanced fertilization for vine health','Monitor irrigation for consistent moisture supply','Keep up preventive spray schedule during key growth stages'],
        'prevention': ['Prune annually during dormant season for open canopy','Maintain good canopy management throughout season','Apply consistent irrigation without wetting foliage','Scout vines regularly and keep records of observations']
    },
    'Orange - Citrus Greening': {
        'severity': 'high',
        'symptoms': ['Asymmetric blotchy yellowing of leaves called huanglongbing pattern','Fruit remains small green and misshapen at harvest','Bitter off-flavor in affected fruit making it unmarketable','Dieback of shoots and branches progressing from top downward','Transmitted by Asian citrus psyllid insect vector'],
        'treatment': ['No cure exists once a tree is infected','Remove and destroy infected trees immediately to prevent spread','Control psyllid vector population with imidacloprid or spinosad','Nutritional sprays may temporarily reduce symptoms but not cure'],
        'prevention': ['Use certified disease-free nursery stock only','Control Asian citrus psyllid populations aggressively','Inspect new trees carefully before introducing to orchard','Report suspected infections to local agricultural authorities immediately']
    },
    'Peach - Bacterial Spot': {
        'severity': 'medium',
        'symptoms': ['Water-soaked lesions on leaves that turn dark and drop out creating shot-holes','Sunken dark spots with cracked centers on fruit surface','Young fruit may crack and drop from tree prematurely','Severe defoliation in susceptible varieties during wet seasons','Stem lesions appear as dark water-soaked streaks'],
        'treatment': ['Apply copper bactericide beginning at bud swell stage','Thin fruit to reduce crowding and improve drying','Avoid high nitrogen fertilization which increases susceptibility','Remove and destroy severely infected plant material'],
        'prevention': ['Plant resistant peach varieties wherever possible','Ensure good air circulation through open canopy pruning','Provide wind protection to reduce spread between trees','Begin copper bactericide sprays early each spring']
    },
    'Peach - Healthy': {
        'severity': 'low',
        'symptoms': ['Healthy green lanceolate leaves with no spots or shot-holes','No signs of gummosis cankers or bacterial lesions','Healthy flower and fruit development each season','Normal peach fragrance from leaves when bruised','Strong shoot growth with typical internode length'],
        'treatment': ['Continue open vase pruning program each dormant season','Thin fruit for improved size quality and disease resistance','Maintain preventive fungicide and bactericide spray schedule','Monitor trunk for gummosis or borer activity regularly'],
        'prevention': ['Maintain annual open-center pruning for good airflow','Avoid wetting foliage with overhead irrigation','Apply timely dormant copper sprays each winter','Inspect trees weekly during the growing season']
    },
    'Pepper - Bacterial Spot': {
        'severity': 'high',
        'symptoms': ['Small water-soaked lesions on leaves becoming brown to black','Yellow halos surrounding dark lesions on leaf surface','Raised scabby lesions on fruit reducing marketability','Premature defoliation exposes fruit to sunscald','High humidity greatly increases disease severity and spread'],
        'treatment': ['Apply copper bactericide beginning at transplanting time','Repeat applications every 5 to 7 days during wet weather','Remove severely infected plants to prevent further spread','Avoid working in the field when foliage is wet'],
        'prevention': ['Use resistant pepper varieties as primary protection','Use drip irrigation only and avoid all overhead watering','Rotate crops for at least 3 years away from peppers','Sanitize all tools and equipment between plants and fields']
    },
    'Pepper - Healthy': {
        'severity': 'low',
        'symptoms': ['Bright green healthy leaves with no spots or lesions','No yellowing wilting or bacterial lesions anywhere','Healthy flower and fruit set progressing normally','Firm crisp fruits developing with good color','No signs of disease pest damage or nutrient deficiency'],
        'treatment': ['Continue regular calcium and boron fertilization for fruit quality','Maintain consistent irrigation to prevent blossom end rot','Monitor for aphids and thrips which can transmit viruses','Stake or cage plants to support vigorous growth'],
        'prevention': ['Use consistent drip irrigation throughout the season','Use certified transplants from a reputable source','Rotate crops annually to prevent soilborne disease buildup','Maintain good air circulation with proper plant spacing']
    },
    'Potato - Early Blight': {
        'severity': 'medium',
        'symptoms': ['Dark brown circular spots with concentric rings creating target appearance','Yellow halo surrounding lesions on leaf surface','Lower leaves are affected first with disease moving upward','Lesions on tubers appear as dark sunken circular patches','Defoliation weakens plants and reduces tuber development'],
        'treatment': ['Apply chlorothalonil fungicide preventively before symptoms appear','Remove infected leaves and dispose away from the field','Avoid excess nitrogen fertilization which increases susceptibility','Harvest at correct maturity to avoid leaving tubers in soil too long'],
        'prevention': ['Use certified disease-free seed potatoes only','Rotate crops on a minimum 3-year cycle','Use drip irrigation and avoid overhead watering','Choose early blight resistant varieties where available']
    },
    'Potato - Late Blight': {
        'severity': 'high',
        'symptoms': ['Large irregular water-soaked lesions on leaves that expand rapidly','White sporulation visible on leaf undersides in humid conditions','Rapid browning and complete collapse of leaf tissue','Dark brown lesions on tubers extending into flesh','Entire plant can collapse within days in severe wet conditions'],
        'treatment': ['Apply mancozeb or metalaxyl fungicide at very first sign','Remove and bag all infected plant material immediately','Do not compost any infected material under any circumstances','Consider early harvest if infection is severe and widespread'],
        'prevention': ['Use certified disease-free seed potatoes only','Avoid all overhead irrigation throughout the season','Plant in well-drained soil to reduce surface moisture','Destroy all crop debris completely after harvest']
    },
    'Potato - Healthy': {
        'severity': 'low',
        'symptoms': ['Healthy dark green foliage with no visible lesions or spots','No signs of water-soaked lesions or white sporulation','Vigorous upright plant growth with normal leaf size','Healthy flower development progressing normally','No signs of blight pest damage or nutrient deficiency'],
        'treatment': ['Continue regular hilling to support tuber development','Maintain consistent soil moisture to prevent growth cracking','Monitor for early signs of blight weekly during season','Apply balanced fertilization avoiding excess nitrogen'],
        'prevention': ['Use certified seed potatoes for every planting','Practice 3-year minimum crop rotation','Hill plants regularly to prevent tuber greening','Scout fields weekly throughout the growing season']
    },
    'Raspberry - Healthy': {
        'severity': 'low',
        'symptoms': ['Healthy green leaves with no spots lesions or yellowing','Vigorous primocane and floricane growth each season','Healthy flower and berry development progressing normally','No signs of cane blight anthracnose or rust','No pest damage or webbing present on plants'],
        'treatment': ['Prune out all spent floricanes after harvest each year','Thin new primocanes to 4 to 6 per hill for airflow','Maintain consistent irrigation without wetting foliage','Apply balanced fertilization in early spring'],
        'prevention': ['Prune annually to maintain open row structure','Use drip irrigation to keep foliage dry','Remove spent canes promptly after harvest each season','Scout plants regularly for any signs of disease or pests']
    },
    'Soybean - Healthy': {
        'severity': 'low',
        'symptoms': ['Uniform green leaf color with no spots yellowing or lesions','Healthy trifoliate leaves with normal size and texture','Vigorous plant growth with normal branching pattern','Healthy pod set and fill progressing normally','No signs of disease pest damage or nutrient deficiency'],
        'treatment': ['Continue balanced fertilization based on soil test results','Maintain adequate soil moisture during pod fill stage','Scout fields weekly for any early signs of disease','Manage weeds promptly to reduce competition and disease hosts'],
        'prevention': ['Rotate crops annually away from soybeans','Plant at optimal population for your variety and soil type','Select varieties with disease resistance packages','Scout fields from emergence through pod fill stage']
    },
    'Squash - Powdery Mildew': {
        'severity': 'medium',
        'symptoms': ['White powdery spots appearing on upper leaf surfaces first','Spots enlarge rapidly to cover entire leaf surface','Severely infected leaves turn yellow then die and dry up','Stems and petioles may also show powdery coating','Late season infections are most common and damaging'],
        'treatment': ['Apply sulfur or potassium bicarbonate at very first sign','Use neem oil as an organic option applied every 7 days','Apply myclobutanil fungicide for established infections','Remove heavily infected leaves to reduce spore load in field'],
        'prevention': ['Ensure good air circulation between plants at all times','Plant resistant varieties as the most effective strategy','Begin preventive sprays at first sign of infection','Avoid excess nitrogen which promotes lush susceptible growth']
    },
    'Strawberry - Leaf Scorch': {
        'severity': 'medium',
        'symptoms': ['Small purple to reddish-purple spots on leaf surfaces','Spots enlarge with centers turning brown then white-gray','Severely infected leaves appear scorched or burned','Spots may merge causing large necrotic areas on leaf','Defoliation reduces plant vigor and yield significantly'],
        'treatment': ['Apply captan or myclobutanil fungicide before wet periods','Remove infected leaves and destroy them away from the field','Improve row drainage to avoid waterlogged conditions','Renovate planting by mowing and removing all debris after harvest'],
        'prevention': ['Use certified disease-free planting stock always','Replace plantings every 3 to 4 years to reset disease pressure','Use drip irrigation to avoid wetting foliage','Maintain good air circulation between rows and plants']
    },
    'Strawberry - Healthy': {
        'severity': 'low',
        'symptoms': ['Bright green trifoliate leaves with no spots or discoloration','No signs of leaf scorch anthracnose or gray mold','Healthy runner and flower production each season','Firm red fruit developing with good size and color','No signs of disease or pest damage anywhere'],
        'treatment': ['Renovate rows by mowing immediately after harvest','Remove excess runners to prevent overcrowding in the row','Apply potassium fertilizer to support fruit quality','Maintain consistent drip irrigation throughout season'],
        'prevention': ['Use certified planting stock from a reputable source','Replace plantings every 3 to 4 years','Apply consistent drip irrigation throughout the season','Remove plant debris thoroughly after each harvest']
    },
    'Tomato - Bacterial Spot': {
        'severity': 'high',
        'symptoms': ['Small water-soaked lesions turning dark brown to black on leaves','Yellow halo surrounding each infected spot on leaf','Raised scab-like spots on fruit surface reducing marketability','Premature defoliation of severely infected plants','Lesions coalesce under high humidity causing leaf blight'],
        'treatment': ['Apply copper hydroxide or copper sulfate every 7 to 10 days','Remove and destroy all visibly infected leaves immediately','Switch to drip irrigation to keep foliage completely dry','Space plants adequately and stake to improve air circulation'],
        'prevention': ['Use certified disease-free transplants and seed','Use drip irrigation only throughout the season','Rotate crops on a 2 to 3 year cycle away from tomatoes','Sanitize all tools between plants to prevent mechanical spread']
    },
    'Tomato - Early Blight': {
        'severity': 'medium',
        'symptoms': ['Dark brown circular spots with concentric rings creating target appearance','Yellow halo surrounds lesions on leaf surface','Lower leaves are affected first with disease progressing upward','Lesions may cause premature defoliation of plants','Dark sunken lesions may appear on stems near the soil line'],
        'treatment': ['Remove affected lower leaves and dispose away from garden','Apply chlorothalonil or mancozeb fungicide every 7 days','Apply mulch around plants to prevent soil splash onto leaves','Fertilize appropriately as stressed plants are most susceptible'],
        'prevention': ['Rotate crops every 2 to 3 years away from tomatoes','Water only at base of plants and never wet foliage','Ensure adequate sunlight and proper plant spacing','Choose early blight resistant varieties where available']
    },
    'Tomato - Late Blight': {
        'severity': 'high',
        'symptoms': ['Large irregular water-soaked lesions on leaves that spread rapidly','White mold visible on leaf undersides in humid conditions','Rapid browning and complete collapse of leaf tissue','Dark brown sunken lesions on fruit surface','Entire plant may collapse within days under severe conditions'],
        'treatment': ['Apply mancozeb or metalaxyl at very first sign of infection','Remove and bag all infected plants immediately','Do not compost any infected material under any circumstances','Switch to drip irrigation and stop all overhead watering'],
        'prevention': ['Use certified disease-free transplants always','Avoid all overhead watering throughout the season','Plant in well-drained soil with good sun exposure','Destroy all crop debris completely after harvest']
    },
    'Tomato - Leaf Mold': {
        'severity': 'medium',
        'symptoms': ['Pale green to yellow spots on upper leaf surface','Olive-gray velvety mold growing on leaf underside','Older and lower leaves are affected first','Leaves curl upward and wither under severe infection','Fruit is rarely affected but plants lose vigor'],
        'treatment': ['Improve greenhouse or tunnel ventilation immediately','Apply copper or chlorothalonil fungicide preventively','Remove and destroy all infected leaves carefully','Reduce plant density by pruning to improve airflow'],
        'prevention': ['Maintain good air circulation in greenhouse or tunnel','Ventilate growing structures during the day','Avoid wetting foliage with any irrigation method','Control humidity below 85 percent to prevent sporulation']
    },
    'Tomato - Septoria Leaf Spot': {
        'severity': 'medium',
        'symptoms': ['Numerous small circular spots with dark borders on leaves','Tan or gray centers with dark brown margins in each spot','Small black dots called pycnidia visible in lesion centers','Lower leaves are infected first with spread moving upward','Severe defoliation occurs during extended wet seasons'],
        'treatment': ['Remove infected lower leaves starting at very first sign','Apply chlorothalonil or mancozeb every 7 to 10 days','Apply mulch on soil surface to prevent spore splash','Avoid all overhead watering to prevent spore germination'],
        'prevention': ['Rotate crops every 2 to 3 years away from tomatoes','Use drip irrigation throughout the growing season','Remove all plant debris after harvest each season','Maintain adequate plant spacing for good air movement']
    },
    'Tomato - Spider Mites': {
        'severity': 'medium',
        'symptoms': ['Fine stippling or bronzing appearing on upper leaf surface','Fine webbing clearly visible on leaf undersides','Leaves turn yellow then bronze and eventually die','Infestation is worst during hot and dry weather conditions','Tiny moving dots visible when looking at leaf undersides'],
        'treatment': ['Apply miticide or insecticidal soap spraying leaf undersides thoroughly','Increase relative humidity around plants as mites hate moisture','Introduce predatory mites such as Phytoseiidae for biocontrol','Remove and destroy heavily infested leaves immediately'],
        'prevention': ['Maintain adequate soil moisture to avoid plant stress','Avoid conditions of heat stress and drought','Encourage natural predator populations in and around the field','Remove crop debris promptly after each harvest']
    },
    'Tomato - Target Spot': {
        'severity': 'medium',
        'symptoms': ['Circular brown spots with concentric rings on leaves and fruit','Yellow halos clearly surrounding each lesion','Spots appear on leaves stems and fruit simultaneously','Lesions may merge causing extensive blight on leaves','Defoliation occurs in severe and prolonged infections'],
        'treatment': ['Apply azoxystrobin or chlorothalonil at early signs of disease','Remove infected tissue by pruning as soon as spotted','Avoid all overhead irrigation to keep foliage dry','Space and stake plants properly to improve air circulation'],
        'prevention': ['Rotate crops away from tomatoes for 2 years minimum','Maintain good air circulation through proper spacing','Use drip irrigation throughout the entire season','Remove all plant debris completely after harvest']
    },
    'Tomato - Yellow Leaf Curl Virus': {
        'severity': 'high',
        'symptoms': ['Severe upward and inward leaf curling on all plant leaves','Leaves turn yellow with green veins remaining in contrast','Stunted plant growth and very poor overall vigor','Significant flower drop leading to very poor fruit set','Virus is transmitted exclusively by the whitefly insect vector'],
        'treatment': ['Control whitefly populations immediately with imidacloprid or neem oil','Remove and destroy all infected plants to prevent spread','Use reflective mulch around plants to repel whitefly vectors','No chemical cure exists so management focuses entirely on vector control'],
        'prevention': ['Plant only virus-resistant tomato varieties','Implement aggressive whitefly control program from planting','Use certified disease-free transplants always','Screen greenhouse openings to exclude whitefly entry']
    },
    'Tomato - Mosaic Virus': {
        'severity': 'high',
        'symptoms': ['Mosaic pattern of light and dark green patches on leaves','Leaf distortion and mottling throughout the plant','Stunted overall plant growth and reduced vigor','Reduced fruit quality size and marketable yield','No insect vector involved as it spreads by contact only'],
        'treatment': ['Remove and destroy all infected plants immediately','Wash hands and all tools thoroughly before handling plants','Use virus-resistant tomato varieties for future plantings','Disinfect all equipment with 10 percent bleach solution'],
        'prevention': ['Plant only virus-resistant varieties','Wash hands before handling any plants in the field','Sterilize all pruning tools between every single plant','Do not smoke near tomato plants as tobacco carries the virus']
    },
    'Tomato - Healthy': {
        'severity': 'low',
        'symptoms': ['Vibrant deep green coloration on all leaves throughout plant','Firm leaf texture with absolutely no spots or lesions','No visible yellowing browning or signs of disease anywhere','Healthy vascular structure visible in leaf midribs','No pest damage webbing or nutrient deficiency signs'],
        'treatment': ['Continue regular watering and maintain consistent soil moisture','Monitor weekly for any early signs of disease or pests','Apply balanced NPK fertilization throughout the season','Maintain proper plant spacing and staking for good airflow'],
        'prevention': ['Water consistently at base of plants and never wet foliage','Ensure full sun exposure for minimum 8 hours daily','Rotate crops each season away from tomatoes and related plants','Sanitize all tools thoroughly between plants and beds']
    },
    'Orange - Citrus Greening': {
        'severity': 'high',
        'symptoms': ['Asymmetric blotchy yellowing of leaves called huanglongbing','Fruit remains small green and misshapen at harvest time','Bitter off-flavor in affected fruit making it unmarketable','Dieback of shoots and branches progressing from top down','Transmitted by Asian citrus psyllid as the insect vector'],
        'treatment': ['No cure exists once a tree is confirmed infected','Remove and destroy infected trees immediately to prevent spread','Control psyllid vector population with imidacloprid or spinosad','Nutritional sprays may reduce symptoms temporarily but not cure the disease'],
        'prevention': ['Use certified disease-free nursery stock only','Control Asian citrus psyllid populations aggressively year-round','Inspect new trees carefully before introducing to orchard','Report all suspected infections to local agricultural authorities immediately']
    },
}

# ====================== BEST LONG-TERM TRANSLATION (JSON CACHE) ======================
print("Loading translations from cache...")

PRE_TRANSLATED = {}

# Try to load from pre-generated cache file
if os.path.exists('translations_cache.json'):
    try:
        with open('translations_cache.json', 'r', encoding='utf-8') as f:
            PRE_TRANSLATED = json.load(f)
        print(f"✅ Loaded translations cache for {len(PRE_TRANSLATED)} diseases (instant)")
    except Exception as e:
        print(f"⚠️  Failed to load cache: {e}")
        PRE_TRANSLATED = {}
else:
    print("⚠️  translations_cache.json not found. Generating on-demand...")

# Built-in English translations (fast fallback, no API needed)
ENGLISH_TRANSLATIONS = {}
for disease_key, info in DISEASE_DB.items():
    is_healthy = 'Healthy' in disease_key
    base_message = 'Healthy leaf detected! No disease found.' if is_healthy else f'Disease detected: {disease_key.replace("___", " - ").replace("_", " ")}'
    ENGLISH_TRANSLATIONS[disease_key] = {
        'disease_name': disease_key.replace('___', ' - ').replace('_', ' '),
        'message': base_message,
        'symptoms': info.get('symptoms', []),
        'treatment': info.get('treatment', []),
        'prevention': info.get('prevention', [])
    }

# Helper to get translations (cached or on-demand)
def get_disease_translations(disease_key):
    """Get translations for a disease. Uses cache if available, otherwise English."""
    if disease_key in PRE_TRANSLATED:
        return PRE_TRANSLATED[disease_key]
    
    # Fallback to English for all languages
    return {
        'en': ENGLISH_TRANSLATIONS.get(disease_key, ENGLISH_TRANSLATIONS.get('default', {})),
        'te': ENGLISH_TRANSLATIONS.get(disease_key, ENGLISH_TRANSLATIONS.get('default', {})),
        'hi': ENGLISH_TRANSLATIONS.get(disease_key, ENGLISH_TRANSLATIONS.get('default', {})),
        'ta': ENGLISH_TRANSLATIONS.get(disease_key, ENGLISH_TRANSLATIONS.get('default', {})),
        'kn': ENGLISH_TRANSLATIONS.get(disease_key, ENGLISH_TRANSLATIONS.get('default', {}))
    }

print(f"✅ Ready! (Using cached translations for {len(PRE_TRANSLATED)} diseases)")
# ==================================================================================

# Default for any class not in the DB
DEFAULT_INFO = {
    'severity': 'medium',
    'symptoms': ['Visible discoloration or lesions on leaf surface','Abnormal spotting or blotching pattern on leaves','Possible yellowing around affected areas','Changes in leaf texture or appearance','Reduced plant vigor and growth'],
    'treatment': ['Isolate affected plants to prevent spread to healthy ones','Remove infected tissue by pruning affected leaves and stems carefully','Apply appropriate fungicide or pesticide after consulting local extension','Improve growing conditions including spacing drainage and nutrition'],
    'prevention': ['Use disease-resistant varieties whenever possible','Water only at base of plants and never wet foliage','Maintain good air circulation with proper plant spacing','Practice crop rotation every 2 to 3 years']
}

LANG_CODES = {
    'te': 'te',
    'hi': 'hi',
    'ta': 'ta',
    'kn': 'kn'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    temp_path = 'temp_upload.jpg'
    file.save(temp_path)

    try:
        # ── Preprocess ──────────────────────────────────────
        img       = image.load_img(temp_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # ── Inference ───────────────────────────────────────
        predictions     = model.predict(img_array, verbose=0)
        predicted_index = int(np.argmax(predictions))
        confidence      = float(predictions[0][predicted_index])
        predicted_class = CLASS_NAMES[predicted_index]

        # ── Top-3 alternatives ──────────────────────────────
        top3_idx  = np.argsort(predictions[0])[::-1][:4]
        alternatives = [
            {
                'label':      CLASS_NAMES[int(i)],
                'confidence': round(float(predictions[0][i]) * 100, 1)
            }
            for i in top3_idx if int(i) != predicted_index
        ][:3]

        # ── Not a Leaf / low confidence ─────────────────────
        if confidence < 0.50 or predicted_class == 'Not a Leaf':
            return jsonify({
                'result':     'Not a Leaf',
                'confidence': round(confidence * 100, 1),
                'is_healthy': False,
                'is_not_leaf': True,
                'en': {'disease_name': 'Not a Leaf', 'message': 'Please upload a clear close-up photo of a plant leaf.', 'symptoms': [], 'treatment': [], 'prevention': []},
                'te': {'disease_name': 'ఆకు కాదు',   'message': 'దయచేసి మొక్క ఆకు యొక్క స్పష్టమైన క్లోజప్ ఫోటోను అప్‌లోడ్ చేయండి.', 'symptoms': [], 'treatment': [], 'prevention': []},
                'hi': {'disease_name': 'पत्ता नहीं',  'message': 'कृपया पौधे की पत्ती की स्पष्ट क्लोज़-अप फ़ोटो अपलोड करें।', 'symptoms': [], 'treatment': [], 'prevention': []},
                'ta': {'disease_name': 'இலை இல்லை',  'message': 'தயவுசெய்து தாவர இலையின் தெளிவான க்ளோஸ்-அப் புகைப்படத்தை பதிவேற்றவும்.', 'symptoms': [], 'treatment': [], 'prevention': []},
                'kn': {'disease_name': 'ಎಲೆ ಅಲ್ಲ',   'message': 'ದಯವಿಟ್ಟು ಸಸ್ಯದ ಎಲೆಯ ಸ್ಪಷ್ಟ ಕ್ಲೋಸ್-ಅಪ್ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.', 'symptoms': [], 'treatment': [], 'prevention': []},
            })

        # ── Get disease info ─────────────────────────────────
        info       = DISEASE_DB.get(predicted_class, DEFAULT_INFO)
        is_healthy = 'Healthy' in predicted_class

        # === FIXED MULTI-LANGUAGE RETURN (Telugu + others will work now) ===
        if predicted_class in PRE_TRANSLATED and PRE_TRANSLATED[predicted_class]:
            trans = PRE_TRANSLATED[predicted_class]
        else:
            trans = PRE_TRANSLATED.get('default', {})

        # Make sure all 5 languages are present (fallback to English if missing)
        result_data = {
            'en': trans.get('en', {}),
            'te': trans.get('te', trans.get('en', {})),
            'hi': trans.get('hi', trans.get('en', {})),
            'ta': trans.get('ta', trans.get('en', {})),
            'kn': trans.get('kn', trans.get('en', {})),
        }

        # Return the full result
        return jsonify({
            'result': predicted_class,
            'confidence': round(confidence * 100, 1),
            'is_healthy': is_healthy,
            'is_not_leaf': False,
            'severity': info['severity'],
            'alternatives': alternatives,
            'en': result_data['en'],
            'te': result_data['te'],
            'hi': result_data['hi'],
            'ta': result_data['ta'],
            'kn': result_data['kn'],
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None, 'classes': len(CLASS_NAMES)})

if __name__ == '__main__':
    # Use the PORT environment variable provided by Render, default to 5000 for local dev
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)