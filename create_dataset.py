import pandas as pd
import random

random.seed(42)

TITLES = ["Mr.", "Ms."]

SURNAMES = {
    "Native": [
        "Begay", "Yazzie", "Benally", "Tsosie", "Nez", "Begaye", "Etsitty", "Becenti",
        "Yellowhair", "Manygoats", "Wauneka", "Manuelito", "Apachito", "Bedonie", "Calabaza",
        "Peshlakai", "Claw", "Roanhorse", "Goldtooth", "Etcitty", "Tsinnijinnie", "Notah",
        "Clah", "Atcitty", "Twobulls", "Werito", "Hosteen", "Yellowman", "Attakai",
        "Bitsui", "Delgarito", "Henio", "Goseyun", "Keams", "Secatero", "Declay", "Tapaha",
        "Beyale", "Haskie", "Cayaditto", "Blackhorse", "Ethelbah", "Tsinnie", "Walkingeagle",
        "Altaha", "Bitsilly", "Wassillie", "Benallie", "Smallcanyon", "Littledog", "Cosay",
        "Clitso", "Tessay", "Secody", "Bigcrow", "Tabaha", "Chasinghawk", "Blueeyes",
        "Olanna", "Blackgoat", "Cowboy", "Kanuho", "Shije", "Gishie", "Littlelight",
        "Laughing", "Whitehat", "Eriacho", "Runningcrane", "Chinana", "Kameroff",
        "Spottedhorse", "Arcoren", "Whiteplume", "Dayzie", "Spottedeagle", "Heavyrunner",
        "Standingrock", "Poorbear", "Ganadonegro", "Ayze", "Whiteface", "Yepa",
        "Talayumptewa", "Madplume", "Bitsuie", "Tsethlikai", "Ahasteen", "Dosela",
        "Birdinground", "Todacheenie", "Bitsie", "Todacheene", "Bullbear", "Lasiloo",
        "Keyonnie", "Notafraid", "Colelay", "Kallestewa", "Littlewhiteman"
    ],
    "Asian": [
        "Nguyen", "Kim", "Patel", "Tran", "Chen", "Li", "Le", "Wang", "Yang", "Pham",
        "Lin", "Liu", "Huang", "Wu", "Zhang", "Shah", "Huynh", "Yu", "Choi", "Ho",
        "Kaur", "Vang", "Chung", "Truong", "Phan", "Xiong", "Lim", "Vo", "Vu", "Lu",
        "Tang", "Cho", "Ngo", "Cheng", "Kang", "Tan", "Ng", "Dang", "Do", "Ly", "Han",
        "Hoang", "Bui", "Sharma", "Chu", "Ma", "Xu", "Zheng", "Song", "Duong", "Liang",
        "Sun", "Zhou", "Thao", "Zhao", "Shin", "Zhu", "Leung", "Hu", "Jiang", "Lai",
        "Gupta", "Cheung", "Desai", "Oh", "Ha", "Cao", "Yi", "Hwang", "Lo", "Dinh",
        "Hsu", "Chau", "Yoon", "Luu", "Trinh", "He", "Her", "Luong", "Mehta", "Moua",
        "Tam", "Ko", "Kwon", "Yoo", "Chiu", "Su", "Shen", "Pan", "Dong", "Begum",
        "Gao", "Guo", "Chowdhury", "Vue", "Thai", "Jain", "Lor", "Yan", "Dao"
    ],
    "Black": [
        "Smalls", "Jeanbaptiste", "Diallo", "Kamara", "Pierrelouis", "Gadson",
        "Jeanlouis", "Bah", "Desir", "Mensah", "Boykins", "Chery", "Jeanpierre",
        "Boateng", "Owusu", "Jama", "Jalloh", "Sesay", "Ndiaye", "Abdullahi",
        "Wigfall", "Bienaime", "Diop", "Edouard", "Toure", "Grandberry", "Fluellen",
        "Manigault", "Abebe", "Sow", "Traore", "Mondesir", "Okafor", "Bangura",
        "Louissaint", "Cisse", "Osei", "Calixte", "Cephas", "Belizaire", "Fofana",
        "Koroma", "Conteh", "Straughter", "Jeancharles", "Mwangi", "Kebede",
        "Mohamud", "Prioleau", "Yeboah", "Appiah", "Ajayi", "Asante", "Filsaime",
        "Hardnett", "Hyppolite", "Saintlouis", "Jeanfrancois", "Ravenell", "Keita",
        "Bekele", "Tadesse", "Mayweather", "Okeke", "Asare", "Ulysse", "Saintil",
        "Tesfaye", "Jeanjacques", "Ojo", "Nwosu", "Okoro", "Fobbs", "Kidane",
        "Petitfrere", "Yohannes", "Warsame", "Lawal", "Desta", "Veasley", "Addo",
        "Leaks", "Gueye", "Mekonnen", "Stfleur", "Balogun", "Adjei", "Opoku",
        "Coaxum", "Vassell", "Prophete", "Lesane", "Metellus", "Exantus", "Hailu",
        "Dorvil", "Frimpong", "Berhane", "Njoroge", "Beyene"
    ],
    "Hispanic": [
        "Garcia", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Perez",
        "Sanchez", "Ramirez", "Torres", "Flores", "Rivera", "Gomez", "Diaz", "Morales",
        "Gutierrez", "Ortiz", "Chavez", "Ruiz", "Alvarez", "Castillo", "Jimenez",
        "Vasquez", "Moreno", "Herrera", "Medina", "Aguilar", "Vargas", "Guzman",
        "Mendez", "Munoz", "Salazar", "Garza", "Soto", "Vazquez", "Alvarado", "Delgado",
        "Pena", "Contreras", "Sandoval", "Guerrero", "Rios", "Estrada", "Ortega",
        "Nunez", "Maldonado", "Dominguez", "Vega", "Espinoza", "Rojas", "Marquez",
        "Padilla", "Mejia", "Juarez", "Figueroa", "Avila", "Molina", "Campos", "Ayala",
        "Carrillo", "Cabrera", "Lara", "Robles", "Cervantes", "Solis", "Salinas",
        "Fuentes", "Velasquez", "Aguirre", "Ochoa", "Cardenas", "Calderon", "Rivas",
        "Serrano", "Rosales", "Castaneda", "Gallegos", "Ibarra", "Suarez", "Orozco",
        "Salas", "Escobar", "Velazquez", "Macias", "Zamora", "Villarreal", "Barrera",
        "Pineda", "Santana", "Trevino", "Lozano", "Rangel", "Arias", "Mora",
        "Valenzuela", "Zuniga", "Melendez", "Galvan", "Velez", "Meza"
    ],
    "White": [
        "Olson", "Snyder", "Wagner", "Meyer", "Schmidt", "Ryan", "Hansen", "Hoffman",
        "Johnston", "Larson", "Carlson", "Obrien", "Jensen", "Hanson", "Weber", "Walsh",
        "Schultz", "Schneider", "Keller", "Beck", "Schwartz", "Becker", "Wolfe",
        "Zimmerman", "Mccarthy", "Erickson", "Klein", "Oconnor", "Swanson",
        "Christensen", "Fischer", "Wolf", "Gallagher", "Schroeder", "Parsons", "Bauer",
        "Mueller", "Hartman", "Kramer", "Flynn", "Owen", "Shaffer", "Hess", "Olsen",
        "Petersen", "Roth", "Hoover", "Weiss", "Decker", "Yoder", "Larsen", "Sweeney",
        "Foley", "Hensley", "Huffman", "Cline", "Oneill", "Koch", "Brennan", "Berg",
        "Russo", "Macdonald", "Kline", "Jacobson", "Berger", "Blankenship", "Bartlett",
        "Odonnell", "Stein", "Stout", "Sexton", "Nielsen", "Howe", "Morse", "Knapp",
        "Herman", "Stark", "Hebert", "Schaefer", "Reilly", "Conrad", "Donovan",
        "Mahoney", "Hahn", "Peck", "Boyle", "Hurley", "Mayer", "Mcmahon", "Case",
        "Duffy", "Friedman", "Fry", "Dougherty", "Crane", "Huber", "Moyer", "Krueger",
        "Rasmussen", "Brandt"
    ]
}

pairs = []
for prop_race, prop_list in SURNAMES.items():
    for prop_surname in prop_list:
        for resp_race, resp_list in SURNAMES.items():
            resp_surname = random.choice(resp_list)
            for prop_title in TITLES:
                for resp_title in TITLES:
                    pairs.append({
                        "proposer_name": f"{prop_title} {prop_surname}",
                        "responder_name": f"{resp_title} {resp_surname}",
                        "proposer_race": prop_race,
                        "responder_race": resp_race
                    })

random.shuffle(pairs)

# Save to a master CSV
df = pd.DataFrame(pairs)
df.to_csv("ultimatum_game-data/experiment_pairs.csv", index=False)
print(f"Saved {len(df)} pairs to experiment_pairs.csv!")