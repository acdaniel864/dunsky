import pandas as pd
import numpy as np
import calendar
from datetime import date, timedelta
import research_lab.utils.database
import time
import re

engine = research_lab.utils.database.get_engine()

def get_wine_advocate_vintage_scores(engine):
    """
    Retrieve Wine Advocate vintage scores from the database.
    """
    
    query = """
    SELECT group_region, sub_group, country, region_label, year, rating_computed, region_names, location_names, color_class_values, date_scraped
    FROM wine_advocate_vintage_scores
    WHERE rating_computed IS NOT NULL;
    """
    
    for attempt in range(2):  # Try twice
        try:
            df_wa = pd.read_sql(query, engine).rename(columns={'color_class_values': 'colour_class_values'}, errors='ignore')
            print(f"Retrieved {len(df_wa)} Wine Advocate vintage scores from database")
            return df_wa
        
        except Exception as e:
            if attempt == 0:
                print(f"First attempt failed retrieving Wine Advocate vintage scores: {e}. Retrying...")
                time.sleep(1)  # Brief pause before retry
            else:
                print(f"Error retrieving Wine Advocate vintage scores after retry: {e}")
                return pd.DataFrame()

def get_wine_spectator_vintage_scores(engine):
    """
    Retrieve Wine Spectator vintage scores from the database.
    
    Args:
        engine: SQLAlchemy engine connection to the database
        
    Returns:
        DataFrame containing Wine Spectator vintage scores
    """
    query = """
    SELECT 
        vintage,
        score,
        drink_window,
        description,
        region
    FROM 
        wine_spectator_vintage_scores
    """
    
    for attempt in range(2):  # Try twice
        try:
            df_ws = pd.read_sql(query, engine)
            print(f"Retrieved {len(df_ws)} Wine Spectator vintage scores from database")
            return df_ws
        
        except Exception as e:
            if attempt == 0:
                print(f"First attempt failed retrieving Wine Spectator vintage scores: {e}. Retrying...")
                time.sleep(1)  # Brief pause before retry
            else:
                print(f"Error retrieving Wine Spectator vintage scores after retry: {e}")
                return pd.DataFrame()


def merge_ws_vintage_scores(df, df_ws, drop_temp_cols=True):
    
    """
    Add wine spectator vintage scores to the main dataframe.
    
    Args:
        df: Main dataframe with wine data
        df_ws: Wine Spectator dataframe with vintage scores
        
    Returns:
        DataFrame with added ws_vintage_score column
    """
    # Clean up wine spectator scores
    def clean_score(score):
        score = str(score).strip()
        
        # Handle score ranges (e.g., "93-96" → 94.5)
        score = score.replace("-", "-").replace("—", "-")  # Normalize dashes
        match = re.match(r"(\d+)-(\d+)", score)
        
        if match:
            low, high = map(int, match.groups())
            return (low + high) / 2  # Midpoint calculation
            
        # Remove * symbols
        score = score.replace("*", "")
        
        return score

    # Clean and process wine spectator scores
    df_ws = df_ws.copy()
    df_ws['score'] = df_ws['score'].apply(clean_score)
    df_ws['score'] = pd.to_numeric(df_ws['score'], errors='coerce')
    df_ws['score'] = df_ws.groupby('region')['score'].transform(lambda x: x.fillna(x.mean()))
    df_ws['score'] = df_ws['score'].astype(float)

    # Dictionary mapping wine regions
    region_subregion_map = {
        'California-cabernet-napa': ['Napa Valley', 'Oakville', 'Rutherford'],
        'California-pinot-noir': [
            'Sonoma Coast', 'Sonoma County', 'Santa Maria Valley', 'Russian River Valley',
            'Sta. Rita Hills', 'Santa Cruz Mountains', 'Anderson Valley', 'Santa Lucia Highlands'
        ],
        'Tuscany-bolgheri-maremma': ['Bolgheri', 'Maremma Toscana'],
        'Tuscany-brunello-di-montalcino': ['Brunello di Montalcino', 'Toscana'],
        'Tuscany-chianti-and-chianti-classico': ['Chianti Classico', 'Colli della Toscana Centrale'],
        'Bordeaux-left-bank-reds-medoc-pessac-leognan': [
            'Pauillac', 'Saint-Julien', 'Margaux', 'Saint-Estephe', 'Haut-Medoc', 'Pessac-Leognan', 
            'Moulis en Medoc', 'Graves', 'Medoc', 'Sauternes', 'Barsac', 'Loupiac'
        ],
        'Bordeaux-right-bank-reds-pomerol-st-emilion': [
            'Saint-Emilion Grand Cru', 'Pomerol', 'Saint-Emilion', 'Saint-Georges-Saint-Emilion', 
            'Castillon-Cotes de Bordeaux', 'Lalande de Pomerol', 'Fronsac', 'Canon-Fronsac', 
            'Montagne-Saint-Emilion', 'Francs-Cotes de Bordeaux', 'Blaye', 'Lussac-Saint-Emilion',
            'Puisseguin-Saint-Emilion', 'Blaye-Cotes de Bordeaux', 'Premieres Cotes de Bordeaux', 
            'Sauternes', 'Barsac', 'Loupiac', 'Graves'
        ],
        'Burgundy-cotes-de-beaune-reds': [
            'Vosne-Romanee', 'Chambolle-Musigny', 'Gevrey-Chambertin', 'Nuits-Saint-Georges', 
            'Clos de Vougeot', 'Volnay', 'Pommard', 'Richebourg', 'Charmes-Chambertin', 
            'Chambertin', 'Echezeaux', 'Mazis-Chambertin', 'Latricieres-Chambertin',  'Monthelie',
            'Pernand-Vergelesses',  
        ],
        'Burgundy-cotes-de-nuits-reds': [
            'Morey-Saint-Denis', 'Chambertin-Clos de Beze', 'Musigny', 'Clos de la Roche', 
            'Chevalier-Montrachet', 'Corton', 'Romanee-Conti', 'La Tache', 'Clos des Lambrays', 
            'Mazoyeres-Chambertin', 'La Grande Rue'
        ],
        # 'Burgundy-older-vintage-reds': [
        # ],
        'Burgundy-white': [
            'Meursault', 'Puligny-Montrachet', 'Chassagne-Montrachet', 'Chablis', 
            'Saint-Aubin',  'Bourgogne Aligote', 'Saint-Romain', 'Petit Chablis', 'Rully', 'Charlemagne', 'Montrachet', 
            'Corton-Charlemagne', 'Batard-Montrachet', 'Criots-Batard-Montrachet'
        ],
        'Rhone-northern': [
            'Cote Rotie', 'Hermitage', 'Cornas', 'Saint-Joseph', 'Condrieu', 'Ermitage', 'Crozes-Hermitage'
        ],
        'Rhone-southern': [
            'Chateauneuf-du-Pape', 'Gigondas', 'Cotes du Rhone', 'Vacqueyras', 'Tavel', 
            'Collines Rhodaniennes', 'Lirac', 'Chateau-Grillet', 'Cairanne', 'Rasteau', 'Ardeche'
        ],
        'Champagne': ['Champagne'],
        'Piedmont': ['Piedmont'],
        'Ribera-del-duero':['Castilla y Leon'], 
        'Rioja': ['Rioja'], 
        "Victoria": [
            "Mornington Peninsula", "Heathcote", "Gippsland", "Beechworth",
            "Strathbogie Ranges", "Rutherglen", "Pyrenees", "Yarra Valley",
            "Geelong", "Macedon Ranges", "Grampians"
        ],
        "Barossa-and-mclaren-vale": ["Barossa Valley", "Eden Valley", "McLaren Vale"],
        "South-africa": ["Klein Karoo", "Breede River Valley", "Cape South Coast", "Western Cape", "Olifants River", "Coastal Region"],
        "New-zealand": ["Auckland","Canterbury", "Central Otago", "Hawke's Bay", "Marlborough", "Nelson", "Wairarapa"],
        "Vintage-port": ["Porto"],
        "Germany": ["Wurttemberg", "Pfalz", "Rheingau", "Mosel", "Baden", "Ahr", "Nahe", "Franken", "Rheinhessen"],
        "Argentina": ["Mendoza", "Patagonia", "Salta"]
    }

    # Create temporary working copy of main df
    df_temp = df.copy()
    
    # Add ws_region column defaulting to 'other'
    df_temp['ws_region'] = 'other'
    
    # Map regions
    for main_region, subregions in region_subregion_map.items():
        for subregion in subregions:
            df_temp.loc[df_temp['region'] == subregion, 'ws_region'] = main_region
            df_temp.loc[df_temp['sub_region'] == subregion, 'ws_region'] = main_region
    
    # Update Burgundy white wines 
    df_temp.loc[(df_temp['region'] == 'Burgundy') & 
                (df_temp['colour'] == 'White'), 'ws_region'] = 'Burgundy-white'
    
    # Update old burgundy red wines
    # First ensure vintage is numeric and filter only where it can be compared
    # Handle 'NV' (non-vintage) and other non-numeric values
    df_temp['vintage_numeric'] = pd.to_numeric(df_temp['vintage'], errors='coerce')

    # Apply the filter only where vintage is a valid number
    df_temp.loc[df_temp['vintage_numeric'].notna() & 
                (df_temp['vintage_numeric'] < 1995) & 
                (df_temp['region'] == 'Burgundy') & 
                (df_temp['colour'] == 'Red'), 'ws_region'] = 'Burgundy-older-vintage-reds'
    
    # Bordeaux old 
    df_temp.loc[df_temp['vintage_numeric'].notna() & 
                (df_temp['vintage_numeric'] < 1995) & 
                (df_temp['region'] == 'Bordeaux') & 
                (df_temp['colour'] == 'Red'), 'ws_region'] = 'Bordeaux-vintage-reds-pre-1995'


    # Create vintage region combinations
    df_temp['ws_region-vintage'] = df_temp['ws_region'] + '-' + df_temp['vintage'].astype(str)
    df_ws['region-vintage'] = df_ws['region'] + '-' + df_ws['vintage'].astype(str)

    # Merge scores
    df_with_ws_scores = pd.merge(
        df_temp,
        df_ws[['region-vintage', 'score']].rename(columns={'score': 'ws_vintage_score'}),
        left_on='ws_region-vintage',
        right_on='region-vintage',
        how='left'
    )
    
    # Fill nulls for Burgundy red wines with mean of Cotes de Nuits and Cotes de Beaune
    burgundy_red_mask = ((df_with_ws_scores['region'] == 'Burgundy') & 
                         (df_with_ws_scores['colour'] == 'Red') & 
                         (df_with_ws_scores['ws_region'] == 'other') & 
                         (df_with_ws_scores['ws_vintage_score'].isna()))
    
    for idx in df_with_ws_scores[burgundy_red_mask].index:
        vintage = df_with_ws_scores.loc[idx, 'vintage']
        beaune_key = f'Burgundy-cotes-de-beaune-reds-{vintage}'
        nuits_key = f'Burgundy-cotes-de-nuits-reds-{vintage}'
        
        beaune_score = df_ws.loc[df_ws['region-vintage'] == beaune_key, 'score'].mean()
        nuits_score = df_ws.loc[df_ws['region-vintage'] == nuits_key, 'score'].mean()
        
        # Calculate mean if both scores exist, otherwise use the one that exists
        if not np.isnan(beaune_score) and not np.isnan(nuits_score):
            df_with_ws_scores.loc[idx, 'ws_vintage_score'] = (beaune_score + nuits_score) / 2
        elif not np.isnan(beaune_score):
            df_with_ws_scores.loc[idx, 'ws_vintage_score'] = beaune_score
        elif not np.isnan(nuits_score):
            df_with_ws_scores.loc[idx, 'ws_vintage_score'] = nuits_score
    
    # Fill nulls for Bordeaux wines with mean of Left Bank and Right Bank
    bordeaux_mask = ((df_with_ws_scores['region'] == 'Bordeaux') & 
                    (df_with_ws_scores['ws_vintage_score'].isna()))
    
    for idx in df_with_ws_scores[bordeaux_mask].index:
        vintage = df_with_ws_scores.loc[idx, 'vintage']
        left_bank_key = f'Bordeaux-left-bank-reds-medoc-pessac-leognan-{vintage}'
        right_bank_key = f'Bordeaux-right-bank-reds-pomerol-st-emilion-{vintage}'
        
        left_bank_score = df_ws.loc[df_ws['region-vintage'] == left_bank_key, 'score'].mean()
        right_bank_score = df_ws.loc[df_ws['region-vintage'] == right_bank_key, 'score'].mean()
        
        # Calculate mean if both scores exist, otherwise use the one that exists
        if not np.isnan(left_bank_score) and not np.isnan(right_bank_score):
            df_with_ws_scores.loc[idx, 'ws_vintage_score'] = (left_bank_score + right_bank_score) / 2
        elif not np.isnan(left_bank_score):
            df_with_ws_scores.loc[idx, 'ws_vintage_score'] = left_bank_score
        elif not np.isnan(right_bank_score):
            df_with_ws_scores.loc[idx, 'ws_vintage_score'] = right_bank_score

    # Drop temporary columns
    if drop_temp_cols:
        df_with_ws_scores = df_with_ws_scores.drop(['ws_region', 'ws_region-vintage', 'region-vintage'], axis=1)
        print(f"Wine Missing WS vintage scores: {df_with_ws_scores['ws_vintage_score'].isna().sum()} ({df_with_ws_scores['ws_vintage_score'].isna().sum() / len(df_with_ws_scores) * 100:.2f}%)")
        return df_with_ws_scores
    else:
        print(f"Wine Missing WS vintage scores: {df_with_ws_scores['ws_vintage_score'].isna().sum()} ({df_with_ws_scores['ws_vintage_score'].isna().sum() / len(df_with_ws_scores) * 100:.2f}%)")

    # Check for wines with multiple WS vintage scores
    duplicate_ws_scores = df_with_ws_scores.groupby('lwin11')['ws_vintage_score'].nunique()
    wines_with_multiple_scores = duplicate_ws_scores[duplicate_ws_scores > 1]
    
    if len(wines_with_multiple_scores) > 0:
        print(f"Found {len(wines_with_multiple_scores)} wines (lwin11s) with multiple WS vintage scores")
        print(f"Example wines with multiple scores: {wines_with_multiple_scores.head().index.tolist()}")
    else:
        print("WS vintage score check: All wines have consistent WS vintage scores")
    
    return df_with_ws_scores

def merge_wine_wa_vintage_scores(df_wine, df_wa_scores):
    """
    Merge wine data with Wine Advocate vintage scores.
    
    Parameters:
    df_wine: DataFrame with wine data including region, sub_region, colour, vintage columns
    df_wa_scores: DataFrame with Wine Advocate scores including region_label, year, rating_computed columns
    
    Returns:
    DataFrame with wine data merged with Wine Advocate vintage scores
    """
    
    def norm(s):
        if s is None:
            return ""
        return re.sub(r"\s+", " ", str(s)).strip().lower()

    def contains_any(text, needles):
        t = norm(text)
        return any(norm(n) in t for n in needles)

    def in_list(val, items):
        return norm(val) in {norm(x) for x in items}
    
    BORDEAUX_SWEET = {"sauternes", "barsac", "loupiac"}

    # Burgundy Côtes (red)
    COTE_DE_NUITS = {
        "gevrey-chambertin", "chambolle-musigny", "vosne-romanee", "nuits-saint-georges",
        "morey-saint-denis", "clos de vougeot", "echezeaux", "mazis-chambertin",
        "latricieres-chambertin", "charmes-chambertin", "mazoyeres-chambertin",
        "la tache", "romanee-conti", "musigny", "la grande rue", "chambertin", "chambertin-clos de beze",
        "clos des lambrays"
    }

    COTE_DE_BEAUNE_RED = {
        "volnay", "pommard", "monthelie", "pernand-vergelesses", "bourgogne"
    }

    # --- Additions for Burgundy communes (safe by colour) ---
    CDN_GRAND_CRU_OR_VILLAGES = {
        "bonnes mares", "clos de la roche", "richebourg", "romanee-saint-vivant",
        "clos saint-denis", "la romanee", "vougeot", "marsannay", "fixin",
        "cote de nuits-villages", "clos de tart"
    }
    CDB_COMMUNES_MIXED = {
        "beaune", "savigny-les-beaune", "santenay", "aloxe-corton", "auxey-duresses", "corton", "meursault"
    }
    # note: 'mercurey' (Côte Chalonnaise) handled separately below (white only)

    # Already present in BURGUNDY_WHITES: 'meursault', 'chassagne-montrachet' (kept as white)

    # Burgundy whites
    BURGUNDY_WHITES = {
        "puligny-montrachet", "chassagne-montrachet", "chablis", "petit chablis",
        "saint-aubin", "saint-romain", "rully", "charlemagne", "montrachet",
        "corton-charlemagne", "batard-montrachet", "criot(s)?-batard-montrachet"
    }

    # Rhône
    RHONE_NORTH = {"cote rotie", "hermitage", "ermitages?", "cornas", "saint-joseph", "condrieu", "crozes-hermitage"}
    RHONE_SOUTH = {"chateauneuf-du-pape", "gigondas", "vacqueyras", "tavel", "cotes du rhone", 
                   "collines rhodaniennes", "lirac", "cairanne", "rasteau", "chateau-grillet", "ardeche"}

    # Italy
    PIEDMONT_BAROLO = {"barolo"}
    PIEDMONT_BARBARESCO = {"barbaresco", "langhe", "barbera d'alba", "dolcetto d'alba", "roero", "nebbiolo d'alba", "barbera d'asti"}
    VENETO_AMARONE = {"amarone"}
    SICILY_ETNA = {"etna"}  # we'll also require Red for "Etna Rosso"
    CAMPANIA_TAURASI = {"taurasi"}
    TUSCANY_BRUNELLO = {"brunello di montalcino"}
    TUSCANY_CHIANTI = {"chianti classico", "chianti", "toscana"}
    TUSCANY_BOLGHERI_SUP = {"bolgheri superiore", "bolgheri"}  # use stricter match if you need exact "Superiore"

    # Spain/Portugal
    RIBERA_DEL_DUERO = {"ribera del duero"}
    PRIORAT = {"priorat"}

    # Germany
    GERMANY_RMN = {"rheingau", "rheinhessen", "pfalz", "franken"}  # Rhein–Main–Neckar umbrella (white)
    GERMANY_MOSEL_NAHE = {"mosel", "nahe"}

    # Austria
    AUSTRIA_NWB = {"niederosterreich", "wien", "burgenland"}
    STYRIA = {"styria", "steiermark"}  # df_wine shows 'Styria (Steiermark)' in categories

    # NZ regions for mapping to country
    NEW_ZEALAND_REGIONS = {"auckland","canterbury","central otago","hawke's bay","marlborough","nelson","wairarapa"}

    # Australia detail
    BAROSSA_AND_MCLAREN = {"barossa valley", "eden valley", "mclaren vale"}

    # US Pacific NW
    # --- Oregon Willamette sub-AVAs ---
    WILLAMETTE = {"willamette valley", "eola-amity hills", "dundee hills"}
    WALLA_WALLA = {"walla walla valley"}

    # California (Santa Barbara / Paso Robles / North Coast subzones)
    SANTA_BARBARA = {"santa barbara", "santa maria valley", "sta. rita hills"}
    PASO_ROBLES = {"paso robles"}
    NORTH_COAST_PINOT_SUBZONES = {"sonoma coast", "sonoma county", "russian river valley", "anderson valley", "santa cruz mountains", "santa lucia highlands"}
    NORTH_COAST_CHARD_SUBZONES = NORTH_COAST_PINOT_SUBZONES  # same geos, use colour to differentiate
    # --- California: broaden North Coast Cab signal (still require Red) ---
    NAPA_SET = {"napa valley", "oakville", "rutherford", "howell mountain", "stags leap district", "mt. veeder", "st. helena"}
    NORTH_COAST_CAB_SIGNALS = NAPA_SET | {"alexander valley", "knights valley"}  # Sonoma/Napa cab hotspots

    # --- California: Carneros (Pinot/Chardonnay) ---
    CARNEROS = {"los carneros"}

    # South Africa buckets
    SOUTH_AFRICA_REGIONS = {"klein karoo", "breede river valley", "cape south coast", "western cape", "olifants river", "coastal region"}

    # Portugal (country-wide dry)
    PORTUGAL_DRY_REGIONS = {
        "dao","alentejo","alentejano","vinho verde","bairrada","douro","minho","tras-os-montes",
        "lisboa","tejo","setubal","tavora-varosa","beira interior","beira atlantico","terrass do dao"
    }

    # Vintage Port signal: in practice, we key on Porto/Douro with sub_region mentioning "port"
    VINTAGE_PORT_HINTS = {"vintage port", "vp", "porto", "port"}


    def map_region_label(row):
        rgn = norm(row.get("region"))
        sub = norm(row.get("sub_region"))
        col = norm(row.get("colour"))

        # Short-circuit simple country/region buckets (strict & safe)
        if rgn == "alsace":
            return "Alsace"
        if rgn == "jura":
            return "Jura"
        if rgn == "champagne":
            return "Champagne"
        if rgn in {"languedoc"}:
            return "Languedoc"
        if rgn in {"roussillon"}:
            return "Roussillon"
        if rgn in {"galicia"}:
            return "Galicia"
        if rgn in {"rioja"}:
            return "Rioja"
        if rgn in {"catalunya"} and contains_any(sub, PRIORAT):
            return "Catalonia: Priorat"
        if rgn in {"england"}:
            # No category for England in your list -> leave unmapped
            return None

        # Portugal: Vintage Port vs dry
        if rgn in {"porto", "douro"} and (contains_any(sub, VINTAGE_PORT_HINTS) or "port" in sub):
            return "Portugal Vintage Port"
        if rgn in PORTUGAL_DRY_REGIONS or rgn in {"porto","douro"}:
            # If it's clearly Port above we already returned; otherwise treat as dry table wine
            return "Portugal dry wines"

        # Australia
        if rgn in {"victoria", "tasmania"}:
            return "Victoria / Tasmania"
        if rgn == "new south wales":
            return "New South Wales"
        if rgn == "western australia":
            return "Western Australia"
        if rgn == "south australia" or contains_any(sub, BAROSSA_AND_MCLAREN):
            return "South Australia: Barossa / McLaren Vale"

        # New Zealand
        if rgn in NEW_ZEALAND_REGIONS or rgn == "new zealand":
            return "New Zealand"

        # Chile / Argentina / South Africa
        if rgn in {"central valley", "aconcagua", "coquimbo"} or rgn == "chile":
            return "Chile"
        if rgn in {"mendoza", "patagonia", "salta"} or rgn == "argentina":
            return "Argentina"
        if rgn in SOUTH_AFRICA_REGIONS or rgn == "south africa":
            return "South Africa"

        # Loire by colour
        if rgn == "loire":
            if col == "red":
                return "Loire Valley (red)"
            if col == "white":
                return "Loire Valley (white)"

        # Germany groupings
        if rgn in GERMANY_MOSEL_NAHE:
            if col == "white":
                return "Mosel and Nahe (white)"
            # Red Mosel/Nahe is rare; leave unmapped rather than mislabel
            return None
        if rgn in GERMANY_RMN:
            if col == "white":
                return "Rhein-Main-Neckar (white)"
            if col == "red":
                # plausible Spätburgunder set
                return "Germany: Pinot Noir / Spätburgunder"

        # Austria: Niederösterreich/Wien/Burgenland (by colour)
        if rgn in AUSTRIA_NWB:
            if col == "white":
                return "Niederösterreich / Wien / Burgenland (white)"
            if col == "red":
                return "Niederösterreich / Wien / Burgenland (red)"
        if in_list(rgn, {"styria"}) or contains_any(rgn, {"steiermark"}):
            return "Styria (Steiermark)"

        # Rhône
        if rgn == "rhone":
            if contains_any(sub, RHONE_NORTH):
                return "Northern Rhône: Cote Rotie / Hermitage"
            # --- Rhône North (already handled via RHONE_NORTH); ensure 'Ermitage' spelling variant works ---
            if contains_any(sub, {"ermitage"}):
                return "Northern Rhône: Cote Rotie / Hermitage"
            if contains_any(sub, RHONE_SOUTH):
                return "Southern Rhône: Chateauneuf du Pape"
            # If subregion missing but region is Rhône, leave unmapped rather than guess.

        # Burgundy (use subregion + colour) - FIXED VERSION
        if rgn == "burgundy" or rgn == "beaujolais" or "bourgogne" in rgn:
            if rgn == "beaujolais":
                return "Burgundy: Beaujolais"
            
            # whites
            if col == "white":
                if contains_any(sub, BURGUNDY_WHITES):
                    return "Burgundy (white)"
                # Check mixed communes for whites
                if contains_any(sub, CDB_COMMUNES_MIXED):
                    return "Burgundy (white)"
                # Generic Bourgogne AOC white
                if norm(row.get("sub_region")) == "bourgogne":
                    return "Burgundy (white)"
                # Mercurey white
                if norm(row.get("sub_region")) == "mercurey":
                    return "Burgundy (white)"
                # Many white AOCs exist; stay conservative:
                return "Burgundy (white)"
            
            # reds: decide Côte
            if col == "red":
                # Check all Côte de Nuits appellations (including Grand Crus like Richebourg)
                if contains_any(sub, COTE_DE_NUITS | CDN_GRAND_CRU_OR_VILLAGES):
                    return "Burgundy: Cote de Nuits (red)"
                # Check Côte de Beaune red appellations
                if contains_any(sub, COTE_DE_BEAUNE_RED):
                    return "Burgundy: Cote de Beaune (red)"
                # Check mixed communes for reds
                if contains_any(sub, CDB_COMMUNES_MIXED):
                    return "Burgundy: Cote de Beaune (red)"
                # If red Burgundy but commune unknown, safest is not to guess Côte:
                return None

        # Piedmont detail
        if rgn == "piedmont" or contains_any(sub, PIEDMONT_BAROLO | PIEDMONT_BARBARESCO):
            if contains_any(sub, PIEDMONT_BAROLO):
                return "Piedmont: Barolo"
            if contains_any(sub, PIEDMONT_BARBARESCO):
                return "Piedmont: Barbaresco"

        # Sicily Etna Rosso
        if rgn == "sicily" and contains_any(sub, SICILY_ETNA) and col == "red":
            return "Sicily: Etna Rosso"

        # Veneto Amarone
        if rgn == "veneto" and (contains_any(sub, VENETO_AMARONE) or "valpolicella" in sub):
            return "Veneto: Amarone della Valpolicella"

        # Campania Taurasi
        if rgn == "campania" and contains_any(sub, CAMPANIA_TAURASI):
            return "Campania: Taurasi"

        # Tuscany
        if rgn == "tuscany" or contains_any(sub, TUSCANY_BRUNELLO | TUSCANY_CHIANTI | TUSCANY_BOLGHERI_SUP):
            if contains_any(sub, TUSCANY_BRUNELLO):
                return "Tuscany: Brunello di Montalcino"
            if contains_any(sub, TUSCANY_CHIANTI):
                return "Tuscany: Chianti Classico"
            if contains_any(sub, TUSCANY_BOLGHERI_SUP):
                # If you want to require explicit 'Superiore' only, tighten this:
                return "Tuscany: Bolgheri Superiore"

        # Spain detail
        if rgn == "castilla y leon" and (contains_any(sub, RIBERA_DEL_DUERO) or "ribera" in sub):
            return "Castilla León: Ribera Del Duero"

        # US West Coast (safe geography-based buckets)
        # --- Oregon: Willamette sub-AVAs (category implies Pinot/Chard, but we only use geography) ---
        if rgn == "oregon" and (contains_any(sub, WILLAMETTE) or "willamette valley" in sub):
            return "Oregon - Willamette Valley: Pinot Noir and Chardonnay"
        if rgn == "washington" and (contains_any(sub, WALLA_WALLA) or col == "red"):
            # category title mentions Cab/Syrah, but we don't have grape; restrict to red and/or Walla Walla
            return "Washington: Cabernet Sauvignon and Syrah"

        # --- California refinements ---
        if rgn == "california":
            # Carneros: Pinot Noir / Chardonnay by colour
            if contains_any(sub, CARNEROS):
                if col == "red":
                    return "California - North Coast: Pinot Noir"
                if col == "white":
                    return "California - North Coast: Chardonnay"

            # North Coast Cabernet Sauvignon: include classic Cab AVAs (only when red)
            if col == "red" and contains_any(sub, NORTH_COAST_CAB_SIGNALS):
                return "California - North Coast: Cabernet Sauvignon"

            if contains_any(sub, SANTA_BARBARA):
                return "California - Central Coast: Santa Barbara"
            if contains_any(sub, PASO_ROBLES):
                return "California - Central Coast: Paso Robles"
            # North Coast varietal categories (use geography + colour as a proxy, still conservative)
            if contains_any(sub, NORTH_COAST_PINOT_SUBZONES) and col == "red":
                return "California - North Coast: Pinot Noir"
            if contains_any(sub, NORTH_COAST_CHARD_SUBZONES) and col == "white":
                return "California - North Coast: Chardonnay"
            # Zinfandel and Cab Sauv (North Coast) require grape to be precise; if Napa subzones + red,
            # you may choose to map to Cab Sauv, but to avoid mistakes, we skip unless you're comfortable:
            if contains_any(sub, NAPA_SET) and col == "red":
                return "California - North Coast: Cabernet Sauvignon"
            # Zinfandel typically Dry Creek/Amador etc., which we do not see in your sample -> leave unmapped.

        if rgn == "bordeaux":
            if contains_any(sub, BORDEAUX_SWEET):
                return "Bordeaux: Barsac / Sauternes"
            if contains_any(sub, {"pessac-leognan", "graves"}):
                return "Bordeaux: Graves / Pessac Leognan"
            if contains_any(sub, {"margaux"}):
                return "Bordeaux: Margaux"
            if contains_any(sub, {"pomerol"}):
                return "Bordeaux: Pomerol"
            if contains_any(sub, {"saint-emilion"}):
                return "Bordeaux: St. Emilion"
            if contains_any(sub, {"saint-julien", "pauillac", "saint-estephe"}):
                return "Bordeaux: St. Julien / Pauillac / St. Estephe"
            if contains_any(sub, {"haut-medoc", "medoc"}):
                return  'Bordeaux: St. Julien / Pauillac / St. Estephe'

        # Default: no safe mapping
        return None

    # --- Apply mapping to df_wine ---------------------------------------------
    df_wine_mapped = df_wine.copy()
    df_wine_mapped["region_label"] = df_wine_mapped.apply(map_region_label, axis=1)

    # --- Merge with df_wa_scores (vintage scores) ---------------------------------------
    # Create composite key for joining
    df_wine_mapped['wa_region_label_year'] = df_wine_mapped['region_label'].astype(str) + '_' + df_wine_mapped['vintage'].astype(str)
    df_wa_scores_copy = df_wa_scores.copy()
    df_wa_scores_copy['wa_region_label_year'] = df_wa_scores_copy['region_label'].astype(str) + '_' + df_wa_scores_copy['year'].astype(str)

    merged_wine_advocate_vintage_scores = df_wine_mapped.merge(
        df_wa_scores_copy.rename(columns={'rating_computed': 'wa_vintage_score'})[['wa_region_label_year', 'wa_vintage_score']], 
        on="wa_region_label_year", 
        how="left")
    
    return merged_wine_advocate_vintage_scores

def add_vintage_year_scores(df_annual):
    
    #TODO: this can be maintained seperately per wine in wine_static with a lambda? every month wine_spectator_vintage_scores refreshes 
    ws_vintage_scores = get_wine_spectator_vintage_scores(engine)

    #TODO: or we can store the matched region in wine spectator table then draw 
    df_with_ws_scores = merge_ws_vintage_scores(df_annual, ws_vintage_scores)

    # Add Wine Advocate vintage scores
    df_wa = get_wine_advocate_vintage_scores(engine)
    
    merged_wine_advocate_vintage_scores = merge_wine_wa_vintage_scores(df_with_ws_scores, df_wa)

    merged_wine_advocate_vintage_scores['mean_vintage_score'] = merged_wine_advocate_vintage_scores[['wa_vintage_score', 'ws_vintage_score']].mean(axis=1)

    merged_wine_advocate_vintage_scores['critic_minus_vintage_score'] = merged_wine_advocate_vintage_scores['average_critic_score'] - merged_wine_advocate_vintage_scores['mean_vintage_score']

    return merged_wine_advocate_vintage_scores