import sys
import re

char_map = {
    # Arabic Letter Hamza
    # "\u": "\u0621",
    
    # Arabic Letter Alef with Hamza Above
    "\uFE83": "\u0623",
    "\uFE84": "\u0623",
    
    # Arabic Letter Yeh with Hamza Above
    "\uFE89": "\u0626",
    "\uFE8A": "\u0626",
    "\uFE8B": "\u0626",
    "\uFE8C": "\u0626",

    # Arabic Letter Waw with Hamza Above
    "\uFE85": "\u0624",
    "\uFE86": "\u0624",
    "\u0676": "\u0624",

    # Arabic Letter Alef with Madda Above
    "\uFE81": "\u0622",  # Arabic letter Alef final form
    "\uFE82": "\u0622",  # Arabic letter Alef isolated form

    # Alef
    "\uFB50": "\u0627",  # Arabic letter Alef wasla
    "\uFE87": "\u0627",
    "\u0675": "\u0627",
    "\u0625": "\u0627",
    "\uFE8D": "\u0627",
    "\uFE8E": "\u0627",
    "\u1EE00": "\u0627",
    "\u1EE80": "\u0627",

    # Beh
    "\uFE8F": "\u0628",
    "\uFE90": "\u0628",
    "\uFE91": "\u0628",
    "\uFE92": "\u0628",
    "\u1EE01": "\u0628",
    "\u1EE21": "\u0628",
    "\u1EE61": "\u0628",
    "\u1EE81": "\u0628",
    "\u1EEA1": "\u0628",

    # Pe
    "\uFB56": "\u067E",
    "\uFB57": "\u067E",
    "\uFB58": "\u067E",
    "\uFB59": "\u067E",

    # Teh
    "\uFE95": "\u062A",
    "\uFE96": "\u062A",
    "\uFE97": "\u062A",
    "\uFE98": "\u062A",
    "\u1EE15": "\u062A",
    "\u1EE35": "\u062A",
    "\u1EE75": "\u062A",
    "\u1EE95": "\u062A",
    "\u1EEB5": "\u062A",

    # Theh
    "\uFE99": "\u062B",
    "\uFE9A": "\u062B",
    "\uFE9B": "\u062B",
    "\uFE9C": "\u062B",
    "\u1EE16": "\u062B",
    "\u1EE36": "\u062B",
    "\u1EE76": "\u062B",
    "\u1EE96": "\u062B",
    "\u1EEB6": "\u062B",

    # Jim
    "\uFE9D": "\u062C",
    "\uFE9E": "\u062C",
    "\uFE9F": "\u062C",
    "\uFEA0": "\u062C",
    "\u1EE02": "\u062C",
    "\u1EE22": "\u062C",
    "\u1EE42": "\u062C",
    "\u1EE62": "\u062C",
    "\u1EE82": "\u062C",
    "\u1EEA2": "\u062C",

    # Cheh
    "\uFB7A": "\u0686",
    "\uFB7B": "\u0686",
    "\uFB7C": "\u0686",
    "\uFB7D": "\u0686",

    # Hah
    "\uFEA1": "\u062D",
    "\uFEA2": "\u062D",
    "\uFEA3": "\u062D",
    "\uFEA4": "\u062D",
    "\u1EE07": "\u062D",
    "\u1EE27": "\u062D",
    "\u1EE47": "\u062D",
    "\u1EE67": "\u062D",
    "\u1EE87": "\u062D",
    "\u1EEA7": "\u062D",

    # Khah
    "\uFEA5": "\u062E",
    "\uFEA6": "\u062E",
    "\uFEA7": "\u062E",
    "\uFEA8": "\u062E",
    "\u1EE17": "\u062E",
    "\u1EE37": "\u062E",
    "\u1EE57": "\u062E",
    "\u1EE77": "\u062E",
    "\u1EE97": "\u062E",
    "\u1EEB7": "\u062E",

    # Dal
    "\uFEA9": "\u062F", 
    "\uFEAA": "\u062F", 
    "\u1EE03": "\u062F", 
    "\u1EE83": "\u062F", 
    "\u1EEA3": "\u062F", 

    # Zal
    "\uFEAB": "\u0630",
    "\uFEAC": "\u0630",
    "\u1EE18": "\u0630",
    "\u1EE98": "\u0630",
    "\u1EEB8": "\u0630",

    # Reh
    "\uFEAE": "\u0631",  # Arabic letter Reh isolated form
    "\uFEAD": "\u0631",  # Arabic letter Reh final form
    "\u0692": "\u0631",
    "\u1EE13": "\u0631",
    "\u1EE93": "\u0631",
    "\u1EEB3": "\u0631",

    # Ze
    "\uFEAF": "\u0632", #
    "\uFEB0": "\u0632", #
    "\u1EE06": "\u0632", #
    "\u1EE86": "\u0632", #
    "\u1EEA6": "\u0632", #

    # Jhe
    "\uFB8A": "\u0698",
    "\uFB8B": "\u0698",

    # Seen
    "\uFEB1": "\u0633", #
    "\uFEB2": "\u0633", #
    "\uFEB3": "\u0633", #
    "\uFEB4": "\u0633", #
    "\u1EE0E": "\u0633", #
    "\u1EE2E": "\u0633", #
    "\u1EE4E": "\u0633", #
    "\u1EE6E": "\u0633", #
    "\u1EE8E": "\u0633", #
    "\u1EEAE": "\u0633", #

    # Sheen
    "\uFEB5": "\u0634", #
    "\uFEB6": "\u0634", #
    "\uFEB7": "\u0634", #
    "\uFEB8": "\u0634", #
    "\u1EE14": "\u0634", #
    "\u1EE34": "\u0634", #
    "\u1EE54": "\u0634", #
    "\u1EE74": "\u0634", #
    "\u1EE94": "\u0634", #
    "\u1EEB4": "\u0634", #

    # Sad
    "\uFEB9": "\u0635", # 
    "\uFEBA": "\u0635", # 
    "\uFEBB": "\u0635", # 
    "\uFEBC": "\u0635", # 
    "\u1EE11": "\u0635", # 
    "\u1EE31": "\u0635", # 
    "\u1EE51": "\u0635", # 
    "\u1EE71": "\u0635", # 
    "\u1EE91": "\u0635", # 
    "\u1EEB1": "\u0635", # 

    # Zad
    "\uFEBD": "\u0636", # 
    "\uFEBE": "\u0636", # 
    "\uFEBF": "\u0636", # 
    "\uFEC0": "\u0636", # 
    "\u1EE19": "\u0636", # 
    "\u1EE39": "\u0636", # 
    "\u1EE59": "\u0636", # 
    "\u1EE79": "\u0636", # 
    "\u1EE99": "\u0636", # 
    "\u1EEB9": "\u0636", # 

    # Ta
    "\uFEC1": "\u0637", #
    "\uFEC2": "\u0637", #
    "\uFEC3": "\u0637", #
    "\uFEC4": "\u0637", #
    "\u1EE08": "\u0637", #
    "\u1EE68": "\u0637", #
    "\u1EE88": "\u0637", #
    "\u1EEA8": "\u0637", #

    # Za
    "\uFEC5": "\u0638", #
    "\uFEC6": "\u0638", #
    "\uFEC7": "\u0638", #
    "\uFEC8": "\u0638", #
    "\u1EE1A": "\u0638", #
    "\u1EE7A": "\u0638", #
    "\u1EE9A": "\u0638", #
    "\u1EEBA": "\u0638", #

    # Ain
    "\uFEC9": "\u0639", # 
    "\uFECA": "\u0639", # 
    "\uFECB": "\u0639", # 
    "\uFECC": "\u0639", # 
    "\u1EE0F": "\u0639", # 
    "\u1EE2F": "\u0639", # 
    "\u1EE4F": "\u0639", # 
    "\u1EE6F": "\u0639", # 
    "\u1EE8F": "\u0639", # 
    "\u1EEAF": "\u0639", # 

    # Ghain
    "\uFECD": "\u063A", #
    "\uFECE": "\u063A", #
    "\uFECF": "\u063A", #
    "\uFED0": "\u063A", #
    "\u1EE1B": "\u063A", #
    "\u1EE3B": "\u063A", #
    "\u1EE5B": "\u063A", #
    "\u1EE7B": "\u063A", #
    "\u1EE9B": "\u063A", #
    "\u1EEBB": "\u063A", #

    # Fa
    "\uFED1": "\u0641", # 
    "\uFED2": "\u0641", # 
    "\uFED3": "\u0641", # 
    "\uFED4": "\u0641", # 
    "\u1EE10": "\u0641", # 
    "\u1EE30": "\u0641", # 
    "\u1EE70": "\u0641", # 
    "\u1EE90": "\u0641", # 
    "\u1EEB0": "\u0641", # 

    # Qaf
    "\uFED5": "\u0642", # 
    "\uFED6": "\u0642", # 
    "\uFED7": "\u0642", # 
    "\uFED8": "\u0642", # 
    "\u1EE12": "\u0642", # 
    "\u1EE32": "\u0642", # 
    "\u1EE52": "\u0642", # 
    "\u1EE72": "\u0642", # 
    "\u1EE92": "\u0642", # 
    "\u1EEB2": "\u0642", # 

    # Kaf
    "\uFB8E": "\u06A9",  # Arabic letter Kaf isolated form
    "\uFB8F": "\u06A9",  # Arabic letter Kaf final form
    "\uFB90": "\u06A9",  # Arabic letter Kaf initial form
    "\uFB91": "\u06A9",  # Arabic letter Kaf medial form
    "\uFCC8": "\u06A9",  # Arabic ligature Dal with Alef final form
    "\u0643": "\u06A9",
    "\uFED9": "\u06A9",
    "\uFEDA": "\u06A9",  # Arabic Letter Kaf Final Form
    "\uFEDB": "\u06A9",  # 
    "\uFEDC": "\u06A9",  # 
    "\u1EE0A": "\u06A9",  # 
    "\u1EE2A": "\u06A9",  # 
    "\u1EE6A": "\u06A9",  # 

    # Gaf
    "\uFB92": "\u06AF",  # Arabic letter Gaf isolated form
    "\uFB93": "\u06AF",  # Arabic letter Gaf final form
    "\uFB94": "\u06AF",  # Arabic letter Gaf initial form
    "\uFB95": "\u06AF",  # Arabic letter Gaf medial form

    # Lam
    "\uFCC9": "\u0644",  # Arabic Ligature Lam with Jeem Initial Form
    "\uFEDD": "\u0644", # Arabic Letter Lam Isolated Form
    "\uFEDE": "\u0644", # Arabic Letter Lam Final Form
    "\uFEDF": "\u0644", # Arabic Letter Lam Initial Form
    "\uFEE0": "\u0644", # Arabic Letter Lam Medial Form
    "\u1EE0B": "\u0644", # Arabic Mathematical Lam
    "\u1EE2B": "\u0644", # Arabic Mathematical Initial Lam
    "\u1EE4B": "\u0644", # Arabic Mathematical Tailed Lam
    "\u1EE8B": "\u0644", # Arabic Mathematical Looped Lam
    "\u1EEAB": "\u0644", # Arabic Mathematical Double-Struck Lam

    # Mim
    "\uFEE1": "\u0645", # Arabic Letter Meem Isolated Form
    "\uFEE2": "\u0645", # Arabic Letter Meem Final Form
    "\uFEE3": "\u0645", # Arabic Letter Meem Initial Form
    "\uFEE4": "\u0645", # Arabic Letter Meem Medial Form
    "\u1EE0C": "\u0645", # Arabic Mathematical Meem
    "\u1EE2C": "\u0645", # Arabic Mathematical Initial Meem
    "\u1EE6C": "\u0645", # Arabic Mathematical Stretched Meem
    "\u1EE8C": "\u0645", # Arabic Mathematical Looped Meem
    "\u1EEAC": "\u0645", # Arabic Mathematical Double-Struck Meem

    # Nun
    "\uFEE5": "\u0646", # Arabic Letter Noon Isolated Form
    "\uFEE6": "\u0646",  # Arabic Letter Noon Final Form
    "\uFEE7": "\u0646",  # Arabic Letter Noon Initial Form
    "\uFEE8": "\u0646",  # Arabic Letter Noon Medial Form
    "\u1EE0D": "\u0646", # Arabic Mathematical Noon
    "\u1EE2D": "\u0646", # Arabic Mathematical Initial Noon
    "\u1EE4D": "\u0646", # Arabic Mathematical Tailed Noon
    "\u1EE6D": "\u0646", # Arabic Mathematical Stretched Noon
    "\u1EE8D": "\u0646", # Arabic Mathematical Looped Noon
    "\u1EEAD": "\u0646", # Arabic Mathematical Double-Struck Noon

    # Vav
    "\u0677": "\u0648",  # Arabic letter Mid hamza on waw
    "\uFEED": "\u0648",  # Arabic Letter Waw Isolated Form
    "\uFEEE": "\u0648",  # Arabic Letter Waw Final Form
    "\u06C6": "\u0648",  # Arabic Letter Oe
    "\u06C7": "\u0648",  # Arabic Letter U

    # He
    "\u06C0": "\u0647",  # Arabic letter Heh with yeh above
    "\u0629": "\u0647",  # Arabic Letter Teh Marbuta
    "\u06BE": "\u0647",  # Arabic Letter Heh Doachashmee
    "\uFE93": "\u0647",  # Arabic Letter Teh Marbuta Isolated Form
    "\u06D5": "\u0647",  # Arabic Letter Ae
    "\uFEE9": "\u0647",  # Arabic Letter Heh Isolated Form
    "\uFEEA": "\u0647",  # Arabic Letter Heh Final Form
    "\uFEEB": "\u0647",  # Arabic Letter Heh Initial Form
    "\uFEEC": "\u0647",  # Arabic Letter Heh Medial Form
    "\u1EE24": "\u0647", # Arabic Mathematical Initial Heh
    "\u1EE64": "\u0647", # Arabic Mathematical Stretched Heh
    "\u1EE84": "\u0647", # Arabic Mathematical Looped Heh

    # Yeh
    "\u06D0": "\u06CC",  # Arabic letter Yeh with dot below
    "\uFEEF": "\u06CC",  # Arabic Letter Alef Maksura Isolated Form
    "\uFEF3": "\u06CC",  # Arabic Letter Yeh Initial Form
    "\uFEF4": "\u06CC",  # Arabic Letter Yeh Medial Form
    "\u064A": "\u06CC",  # Arabic Letter Yeh
    "\uFEF1": "\u06CC",  # Arabic Letter Yeh Isolated Form
    "\u06CE": "\u06CC",  # Arabic Letter Yeh with Small V
    "\uFBFD": "\u06CC",  # Arabic Letter Farsi Yeh Final Form
    "\uFBFC": "\u06CC",  # Arabic Letter Farsi Yeh Isolated Form
    "\uFBFE": "\u06CC",  # Arabic Letter Farsi Yeh Initial Form
    "\uFBFF": "\u06CC",  # Arabic Letter Farsi Yeh Medial Form
    "\uFEF0": "\u06CC",  # Arabic letter Lam final form
    "\uFEF2": "\u06CC",  # Arabic letter Lam medial form
    "\u063D": "\u06CC",
    "\u063E": "\u06CC",
    "\u063F": "\u06CC",
    "\u06D2": "\u06CC", # Arabic Letter Yeh Barree

    "\u064E": "",
    "\u064B": "",
    "\u064F": "",
    "\u064C": "",
    "\u0650": "",
    "\u064D": "",
    "\u0652": "",
    "\u0651": "",
    "\u0654": "",

    "0": "۰",
    "1": "۱",
    "2": "۲",
    "3": "۳",
    "4": "۴",
    "5": "۵",
    "6": "۶",
    "7": "۷",
    "8": "۸",
    "9": "۹",
    "٠": "۰",
    "١": "۱",
    "٢": "۲",
    "٣": "۳",
    "٤": "۴",
    "٥": "۵",
    "٦": "۶",
    "٧": "۷",
    "٨": "۸",
    "٩": "۹",

    "٬": "،",
    ",": "،",
    ";": "؛",
    "?": "؟",
    "\\": " ",
    "…": " غیره ",
    "%": " درصد ",
    "\u200e": " ",# LEFT-TO-RIGHT
    "\u200f": " ",# RIGHT-TO-LEFT
    "\u202a": " ",# LEFT-TO-RIGHT EMBEDDING
    "\u202b": " ",# RIGHT-TO-LEFT EMBEDDING
    "\u2066": " ",# LEFT-TO-RIGHT ISOLATE
    "\u2067": " ",# RIGHT-TO-LEFT ISOLATE
    "\u2069": " ",# POP DIRECTIONAL ISOLATE
    "\ufdef": " ",# Non-standard
    "\u00B7": ".",# MIDDLE DOT
    "\u2022": " ",# BULLET POINT

    "'": " ",
    "“": " ",
    "”": " ",
    "\u00ad": " ",
    "\u005f": " ",
    "\u002b": " ",
    "\u200b": " ",
    # ©
    "\u00a9": " ",

    "\u2014": " ",# Em Dash
    "\u2019": " ",# Right Single Quotation Mark
    "\uFE0F": "",# Variation Selector-16 (VS16)
    "\u007C": " ",# Vertical Line
}

patterns = {
    "SPLIT_.": re.compile("(?<![۰-۹])\\.|\\.(?![۰-۹])"),
    "SPLIT_!": re.compile("!"),
    "SPLIT_؟": re.compile("؟"),
    "ELIMINATE": re.compile("[\\(\\):'\"\\[\\]–ـ«»\\-><]"),
    "COMMA": re.compile("\u060c"),
    "TOO_MANY_SPACES": re.compile("\\s+"),
    "NEG_FARSI_CHARS": re.compile(".*[^\u060c\u061b\u061f\u0622\u0623\u0624\u0626\u0627"
                                  "\u0628\u062a\u062b\u062c\u062d\u062e\u062f\u0630\u0631"
                                  "\u0632\u0633\u0634\u0635\u0636\u0637\u0638\u0639\u063a"
                                  "\u0641\u0642\u0644\u0645\u0646\u0647\u0648\u067e\u0686"
                                  "\u0698\u06a9\u06af\u06cc\u06f0\u06f1\u06f2\u06f3\u06f4"
                                  "\u06f5\u06f6\u06f7\u06f8\u06f9\\s\u200c\\.\\!].*"),
    "NO_SPACE_BEFORE": re.compile("\\s(?=[\\.!؟،])"),
}


def split_into_sentences(paragraph):
    def flat_map(f, xs):
        ys = []
        for x in xs:
            ys.extend(f(x))
        return ys

    def split_strip_append(string, regex_name, ending):
        sentences = [s.strip() for s in
                     patterns[regex_name].split(string)]
        sentences = [f"{s}{ending}" if i != len(sentences) - 1 else s
                     for i, s in enumerate(sentences) if len(s) > 0]
        return sentences

    sentences = split_strip_append(paragraph, "SPLIT_.", ".")
    sentences = flat_map(lambda s: split_strip_append(s, "SPLIT_!", "!"),
                         sentences)
    sentences = flat_map(lambda s: split_strip_append(s, "SPLIT_؟", "؟"),
                         sentences)

    if len(sentences) > 0 and sentences[-1][-1] not in [".", "!", "؟"]:
        sentences[-1] = f"{sentences[-1]}."

    return sentences


def map_to_persian(char):
    if char in char_map:
        return char_map[char]
    else:
        return char


def more_alphabet_than_non(string):
    farsi_characters_count = 0
    farsi_characters = "\u0622\u0623\u0624\u0626\u0627\u0628\u062a\u062b\u062c" \
                       "\u062d\u062e\u062f\u0630\u0631\u0632\u0633\u0634\u0635" \
                       "\u0636\u0637\u0638\u0639\u063a\u0641\u0642\u0644\u0645" \
                       "\u0646\u0647\u0648\u067e\u0686\u0698\u06a9\u06af\u06cc "

    for char in string:
        if char in farsi_characters:
            farsi_characters_count += 1

    return len(string)-farsi_characters_count < farsi_characters_count

