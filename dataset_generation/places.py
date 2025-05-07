def generate_places(place_name):
    #### Oxford all
    if place_name == 'oxford':
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
    return places
