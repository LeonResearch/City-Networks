#!/usr/bin/env python
# coding: utf-8

utrecht = [
    "Amersfoort, Utrecht, The Netherlands",
    "Baarn, Utrecht, The Netherlands",
    "Bunnik, Utrecht, The Netherlands",
    "Bunschoten, Utrecht, The Netherlands",
    "De Bilt, Utrecht, The Netherlands",
    "De Ronde Venen, Utrecht, The Netherlands",
    "Eemnes, Utrecht, The Netherlands",
    "Houten, Utrecht, The Netherlands",
    "IJsselstein, Utrecht, The Netherlands",
    "Leusden, Utrecht, The Netherlands",
    "Lopik, Utrecht, The Netherlands",
    "Montfoort, Utrecht, The Netherlands",
    "Nieuwegein, Utrecht, The Netherlands",
    "Oudewater, Utrecht, The Netherlands",
    "Renswoude, Utrecht, The Netherlands",
    "Rhenen, Utrecht, The Netherlands",
    "Soest, Utrecht, The Netherlands",
    "Stichtse Vecht, Utrecht, The Netherlands",
    "Utrecht, Utrecht, The Netherlands",
    "Utrechtse Heuvelrug, Utrecht, The Netherlands",
    "Veenendaal, Utrecht, The Netherlands",
    "Vijfheerenlanden, Utrecht, The Netherlands",
    "Wijk bij Duurstede, Utrecht, The Netherlands",
    "Woerden, Utrecht, The Netherlands",
    "Zeist, Utrecht, The Netherlands"
]

groningen = [
    "Groningen, Groningen, The Netherlands",
    "Het Hogeland, Groningen, The Netherlands",
    "Midden-Groningen, Groningen, The Netherlands",
    "Oldambt, Groningen, The Netherlands",
    "Pekela, Groningen, The Netherlands",
    "Stadskanaal, Groningen, The Netherlands",
    "Veendam, Groningen, The Netherlands",
    "Westerkwartier, Groningen, The Netherlands",
    "Westerwolde, Groningen, The Netherlands",
    "Eemsdelta, Groningen, The Netherlands"
]

netherlands = [
    "Drenthe, The Netherlands",
    "Flevoland, The Netherlands",
    "Friesland, The Netherlands",
    "Gelderland, The Netherlands",
    "Limburg, The Netherlands",
    "North Brabant, The Netherlands",
    "North Holland, The Netherlands",
    "Overijssel, The Netherlands",
    "South Holland, The Netherlands",
    "Zeeland, The Netherlands"
] + utrecht + groningen

spain = [
    "Andalusia, Spain",
    "Aragon, Spain",
    "Asturias, Spain",
    "Balearic Islands, Spain",
    "Canary Islands, Spain",
    "Cantabria, Spain",
    "Castile and Leon, Spain",
    "Castile-La Mancha, Spain",
    "Catalonia, Spain",
    "Community of Madrid, Spain",
    "Valencian Community, Spain",
    "Extremadura, Spain",
    "Galicia, Spain",
    "Community of Murcia, Spain",
    "Navarre, Spain",
    "Basque Country, Spain",
    "La Rioja, Spain"
]

def generate_places(place_name):
    data_dir = f'./road_data_2025/{place_name}/'
    #### USA west coast & some mountain states
    if place_name == 'usa_west':
        places = [
            "Washington State, USA",
            "Oregon, USA",
            "Nevada, USA",
            "California, USA",
            "Idaho, USA",
            "Utah, USA",
            "Arizona, USA",
        ]
    #### USA estern & central time zone all
    elif place_name == 'usa_east_central':
        places = [
            "Connecticut, USA",
            "Delaware, USA",
            "Florida, USA",
            "Georgia, USA",
            "Indiana, USA",
            "Kentucky, USA",
            "Maine, USA",
            "Maryland, USA",
            "Massachusetts, USA",
            "Michigan, USA",
            "New Hampshire, USA",
            "New Jersey, USA",
            "New York, USA",
            "North Carolina, USA",
            "Ohio, USA",
            "Pennsylvania, USA",
            "Rhode Island, USA",
            "South Carolina, USA",
            "Tennessee, USA",
            "Vermont, USA",
            "Virginia, USA",
            "West Virginia, USA",
            "Alabama, USA",
            "Arkansas, USA",
            "Illinois, USA",
            "Iowa, USA",
            "Kansas, USA",
            "Louisiana, USA",
            "Minnesota, USA",
            "Mississippi, USA",
            "Missouri, USA",
            "Nebraska, USA",
            "North Dakota, USA",
            "Oklahoma, USA",
            "South Dakota, USA",
            "Texas, USA",
            "Wisconsin, USA",
        ]
    #### USA main land 48 states & Washington D.C.
    elif place_name == 'usa_main':
        places = [
            "Alabama, USA",
            "Arizona, USA",
            "Arkansas, USA",
            "California, USA",
            "Colorado, USA",
            "Connecticut, USA",
            "Delaware, USA",
            "District of Columbia, USA",
            "Florida, USA",
            "Georgia, USA",
            "Idaho, USA",
            "Illinois, USA",
            "Indiana, USA",
            "Iowa, USA",
            "Kansas, USA",
            "Kentucky, USA",
            "Louisiana, USA",
            "Maine, USA",
            "Maryland, USA",
            "Massachusetts, USA",
            "Michigan, USA",
            "Minnesota, USA",
            "Mississippi, USA",
            "Missouri, USA",
            "Montana, USA",
            "Nebraska, USA",
            "Nevada, USA",
            "New Hampshire, USA",
            "New Jersey, USA",
            "New Mexico, USA",
            "New York State, USA",
            "North Carolina, USA",
            "North Dakota, USA",
            "Ohio, USA",
            "Oklahoma, USA",
            "Oregon, USA",
            "Pennsylvania, USA",
            "Rhode Island, USA",
            "South Carolina, USA",
            "South Dakota, USA",
            "Tennessee, USA",
            "Texas, USA",
            "Utah, USA",
            "Vermont, USA",
            "Virginia, USA",
            "Washington, USA",
            "West Virginia, USA",
            "Wisconsin, USA",
            "Wyoming, USA",
        ]
    #### Europe west
    elif place_name == 'europe_west':
        places = [
            "Andorra",
            "Austria",
            "Belgium",
            "Metropolitan France",
            "Germany",
            "Italy",
            "Liechtenstein",
            "Luxembourg",
            "Monaco",
            "Continental Portugal",
            "San Marino",
            "Switzerland",
            "Vatican City",
        ] \
        + netherlands \
        + spain 
    #### Europe mainland all
    elif place_name == 'europe_all':
        places = [
            "Albania",
            "Andorra",
            "Austria",
            "Belarus",
            "Belgium",
            "Bosnia and Herzegovina",
            "Bulgaria",
            "Croatia",
            "Czech Republic",
            "Denmark",
            "Estonia",
            "Finland",
            "Metropolitan France",
            "Germany",
            "Greece",
            "Hungary",
            "Iceland",
            "Ireland",
            "Italy",
            "Kosovo",
            "Latvia",
            "Liechtenstein",
            "Lithuania",
            "Luxembourg",
            "Malta",
            "Moldova",
            "Monaco",
            "Montenegro",
            "North Macedonia",
            "Norway",
            "Poland",
            "Continental Portugal",
            "Romania",
            "San Marino",
            "Serbia",
            "Slovakia",
            "Slovenia",
            "Peninsular Spain",
            "Sweden",
            "Switzerland",
            "Ukraine",
            #"United Kingdom",
            "Vatican City",
        ]
    #### Spain all
    elif place_name == 'spain':
        places = spain
    #### Netherlands all
    elif place_name == 'netherlands':
        places = netherlands
    #### UK all
    elif place_name == 'uk':
        places = [
            "England, United Kingdom",
            "Scotland, United Kingdom",
            "Wales, United Kingdom",
            "Northern Ireland, United Kingdom",
        ]
    #### England all
    elif place_name == 'england':
        places = [
            "England, United Kingdom",
        ]
    #### Oxford all
    elif place_name == 'oxford':
        places = [
            "Oxford, England, UK"
        ]
    #### London all
    elif place_name == 'london':
        places = [
            "London, England, UK",
            "City of London, London, England, UK"
        ]
    elif place_name == 'westminster':
        places = [
            "City of Westminster, London, England, UK",
        ]
    #### Sydney all
    elif place_name == 'sydney':
        places = [
            "Sydney, New South Wales, Australia"
        ]
    #### Paris all
    elif place_name == 'paris':
        places = [
            "Paris, France"
        ]
    elif place_name == 'paris_centre':
        places = [
            "1st arrondissement, Paris, France",
            "2nd arrondissement, Paris, France",
            "3rd arrondissement, Paris, France",
            "4th arrondissement, Paris, France",
        ]
    #### New York all
    elif place_name == 'nyc':
        places = [
            "New York City, New York State, USA"
        ]
    #### Toronto all
    elif place_name == 'toronto':
        places = [
            "Toronto, Ontario, Canada"
        ]
    #### L.A. all
    elif place_name == 'la':
        places = [
            "Los Angeles, California, USA"
        ] 
    elif place_name == "pasadena":
        places = [
            "Pasadena, Los Angeles, California, USA"
        ]
    #### Shanghai all
    elif place_name == 'shanghai':
        places = [
            "Baoshan, Shanghai, China",
            "Changning, Shanghai, China",
            "Chongming, Shanghai, China",
            "Fengxian, Shanghai, China",
            "Hongkou, Shanghai, China",
            "Huangpu, Shanghai, China",
            "Jing'an, Shanghai, China",
            "Jiading, Shanghai, China",
            "Jinshan, Shanghai, China",
            "Minhang, Shanghai, China",
            "Pudong, Shanghai, China",
            "Putuo, Shanghai, China",
            "Qingpu, Shanghai, China",
            "Songjiang, Shanghai, China",
            "Xuhui, Shanghai, China",
            "Yangpu, Shanghai, China",
            "Zhabei, Shanghai, China",
        ]
    elif place_name == "huangpu":
        places = [
            "Huangpu, Shanghai, China"
        ]
    return data_dir, places
