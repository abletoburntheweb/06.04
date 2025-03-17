import json
import random

def load_songs(filename="songs.json"):
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)

def generate_playlist(songs, genre_preference, mood, time_of_day):
    playlist = []

    for song in songs:
        if time_of_day == "утро" and mood == "бодрое" and song["genre"] in ["рок", "поп", "электронная"]:
            playlist.append(song)
        elif time_of_day == "вечер" and mood == "спокойное" and song["genre"] in ["джаз", "классика", "лоуфай"]:
            playlist.append(song)
        elif song["genre"] == genre_preference:
            playlist.append(song)
        elif song["popularity"] > 80 and not genre_preference:
            playlist.append(song)

    if not playlist:
        playlist = random.sample(songs, min(3, len(songs)))

    return playlist

songs = load_songs()

genre = input("Введите ваш любимый жанр (рок, рэп, альтернатива, поп, классика, джаз) или оставьте пустым: ").strip().lower()
mood = input("Введите ваше настроение (бодрое, спокойное): ").strip().lower()
time_of_day = input("Введите время суток (утро, день, вечер): ").strip().lower()

playlist = generate_playlist(songs, genre, mood, time_of_day)

print("\nСгенерированный плейлист:")
for song in playlist:
    print(f"- {song['title']} ({song['artist']}) [{song['genre']}]")
