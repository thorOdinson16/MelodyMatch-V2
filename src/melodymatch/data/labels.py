GENRES = [
    "Blues",
    "Classical",
    "Country",
    "Easy Listening",
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Instrumental",
    "International",
    "Jazz",
    "Old-Time / Historic",
    "Pop",
    "Rock",
    "Soul-RnB",
    "Spoken",
]

GENRE_TO_INDEX = {
    genre: index
    for index, genre in enumerate(GENRES)
}

INDEX_TO_GENRE = {
    index: genre
    for genre, index in GENRE_TO_INDEX.items()
}