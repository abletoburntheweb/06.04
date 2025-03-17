import json

def load_data(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)

def get_recommendations(data, genre, decade):
    decade_prefix = decade[:4]
    for item in data["music_recommendations"]:
        if item["genre"].lower() == genre.lower() and item["decade"] == decade_prefix:
            return item["artists"]
    return None

def show_genre_menu():
    genres = [
        "Рок", "Рэп", "Альтернатива", "Поп", "Электронная",
        "Матрок", "Поп-панк", "Шугейз", "Металкор"
    ]
    print("Выберите жанр: ", ", ".join([f"{i+1}. {genre}" for i, genre in enumerate(genres)]))
    choice = int(input("Введите номер жанра: "))
    return genres[choice - 1]

def show_decade_menu():
    decades = [
        "90-е", "2000-е", "2010-е", "2020-е"
    ]
    print("Выберите десятилетие: ", ", ".join([f"{i+1}. {decade}" for i, decade in enumerate(decades)]))
    choice = int(input("Введите номер десятилетия: "))
    return decades[choice - 1]

def main():
    filename = "music_recommendations.json"
    data = load_data(filename)

    genre = show_genre_menu()
    decade = show_decade_menu()

    recommendations = get_recommendations(data, genre, decade)

    if recommendations is None:
        print("Нет рекомендаций по заданным параметрам.")
    else:
        print("Рекомендованные исполнители:", ", ".join(recommendations))

if __name__ == "__main__":
    main()
