animal(lion, carnivore, savanna, has_tail).
animal(giraffe, herbivore, savanna, has_tail).
animal(elephant, herbivore, savanna, has_tail).
animal(tiger, carnivore, forest, has_tail).
animal(bear, omnivore, forest, has_tail).
animal(kangaroo, herbivore, desert, has_tail).
animal(penguin, carnivore, water, no_tail).
animal(dolphin, carnivore, water, has_fin).
animal(monkey, omnivore, forest, has_tail).
animal(crocodile, carnivore, water, has_tail).

carnivore(lion).
carnivore(tiger).
carnivore(penguin).
carnivore(dolphin).
carnivore(crocodile).

herbivore(giraffe).
herbivore(elephant).
herbivore(kangaroo).

omnivore(bear).
omnivore(monkey).

savanna(lion).
savanna(giraffe).
savanna(elephant).

forest(tiger).
forest(bear).
forest(monkey).

desert(kangaroo).

water(penguin).
water(dolphin).
water(crocodile).

has_tail(lion).
has_tail(giraffe).
has_tail(elephant).
has_tail(tiger).
has_tail(bear).
has_tail(kangaroo).
has_tail(monkey).
has_tail(crocodile).

no_tail(penguin).

has_fin(dolphin).

herbivorous_animal(X) :- animal(X, herbivore, _, _).
carnivorous_animal(X) :- animal(X, carnivore, _, _).
omnivorous_animal(X) :- animal(X, omnivore, _, _).
savana_animal(X) :- animal(X, _, savanna, _).
forest_animal(X) :- animal(X, _, forest, _).
desert_animal(X) :- animal(X, _, desert, _).
water_animal(X) :- animal(X, _, water, _).
animal_with_tail(X) :- animal(X, _, _, has_tail).
animal_without_tail(X) :- animal(X, _, _, no_tail).
animal_with_fin(X) :- animal(X, _, _, has_fin).