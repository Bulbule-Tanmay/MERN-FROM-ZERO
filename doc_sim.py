from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
load_dotenv()
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

documents = [
    "Virat Kohli is an Indian international cricketer and former captain of the Indian national team. Widely regarded as one of the greatest batsmen in modern cricket history, he is renowned for his intense competitiveness, flawless chase mechanics in limited-overs formats, and exceptional fitness standards. Under his leadership, India achieved significant test victories abroad and established a dominant fast-bowling attack.",
    "Mahendra Singh Dhoni, affectionately known as MS Dhoni, is a former captain of the Indian national cricket team who led India to victory in all three major ICC trophies: the 2007 T20 World Cup, the 2011 Cricket World Cup, and the 2013 Champions Trophy. Famous for his lightning-fast stumpings, unshakeable composure under high pressure, and tactical genius as a master finisher, he remains one of the most beloved figures in sports history.",
    "Sachin Tendulkar, often revered by fans as the 'God of Cricket', enjoyed an illustrious international career spanning 24 years. As the all-time leading run-scorer in both Test and ODI cricket and the only player to score a hundred international centuries, his masterly technique, iconic straight drive, and global influence inspired generations of aspiring cricketers around the world.",
    "Rohit Sharma, current captain of the Indian cricket team, is celebrated globally for his effortless batting style, impeccable timing, and unmatched ability to hit big sixes. He holds the world record for the highest individual score in an ODI match (264 runs) and is the only cricketer to have scored three double centuries in the One Day International format.",
    "Jasprit Bumrah is a premier Indian fast bowler considered one of the most lethal death-bowling specialists in modern cricket. Characterized by a unique, highly unorthodox bowling action and a short run-up, he possesses the uncanny ability to bowl searing yorkers, deceptive slower balls, and sharp bouncers consistently across all three international formats.",
    "Arijit Singh is a prominent Indian playback singer and music composer who has dominated the Bollywood music scene for over a decade. Known for his deeply emotional vocal tone, versatile range, and soulful renditions, his tracks frequently top streaming charts and have made him a defining voice of contemporary Indian romantic and acoustic ballads.",
    "Lata Mangeshkar, revered as the 'Nightingale of India', was an iconic playback singer whose extraordinary career spanned more than seven decades. Recording thousands of songs across dozens of regional Indian languages, her classic voice, perfect pitch, and profound emotional depth earned her the Bharat Ratna, India's highest civilian honor.",
    "Arpit Bala is a popular Indian digital content creator, comedian, and streamer known for his surreal humor, satirical sketches, and commentary videos. Building a dedicated online fanbase through live streams and collaborative internet culture, his distinctive personality and comedic timing have made him a recognizable figure among youth audiences.",
    "Amitabh Bachchan, often referred to as the 'Shahenshah' or 'Star of the Millennium', is a legendary figure in Indian cinema. Rising to stardom in the 1970s as the iconic 'Angry Young Man', his career spans over five decades with dozens of award-winning performances, iconic baritone voice roles, and a historic tenure as a television host.",
    "Dr. A.P.J. Abdul Kalam, known as the 'Missile Man of India', was an eminent aerospace scientist who played a pivotal role in developing India's civilian space program and military missile capabilities. Serving as the 11th President of India, he was widely loved for his humility, vision for national development, and dedication to inspiring young students.",
    "Ratan Tata was an esteemed Indian industrialist, philanthropist, and former chairman of the Tata Group, who transformed the conglomerate into a global powerhouse. Renowned for his ethical business leadership, vision for accessible innovation like the Tata Nano, and vast philanthropic contributions, he stands as a symbol of integrity and nation-building.",
    "Neeraj Chopra made history as the first Indian track and field athlete to win an Olympic gold medal, achieving the feat in the javelin throw at the Tokyo Olympics. Known for his explosive power, technical precision, and humble attitude, he brought field sports into national prominence across India.",
    "A.R. Rahman is an internationally acclaimed Indian composer, singer, and music producer known for integrating Eastern classical music with modern electronic and orchestral arrangements. A two-time Academy Award winner for his iconic soundtrack in 'Slumdog Millionaire', he revolutionized the Indian film music industry.",
    "Shah Rukh Khan, often hailed as 'King Khan' or the 'Badshah of Bollywood', is a global cinematic icon and film producer. With a career spanning over three decades, his charismatic screen presence, romantic roles, and entrepreneurial ventures have made him one of the most famous and influential actors in global cinema history.",
    "CarryMinati, whose real name is Ajey Nagar, is a pioneering Indian YouTuber and content creator known for his energetic roasts, satirical gaming commentary, and viral diss tracks. As one of the most subscribed individual creators in Asia, he played a massive role in shaping modern Indian internet culture."
]

doc = embeddings.embed_documents(documents)
q = ""
while q != "exit":
    q = input("Enter your query: ")

    qc = embeddings.embed_query(q)


    cosine_similarities = cosine_similarity([qc], doc)




    print("Most similar document: ", documents[np.argmax(cosine_similarities)])