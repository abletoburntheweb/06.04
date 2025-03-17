male(abraham).
male(clancy).
male(herb).
male(homer).
male(bart).

female(mona).
female(jackie).
female(marge).
female(patty).
female(selma).
female(lisa).
female(maggie).
female(ling).

parent(abraham, herb).
parent(abraham, homer).
parent(mona, homer).
parent(clancy, marge).
parent(clancy, patty).
parent(clancy, selma).
parent(jackie, marge).
parent(jackie, patty).
parent(jackie, selma).
parent(homer, bart).
parent(homer, lisa).
parent(homer, maggie).
parent(marge, bart).
parent(marge, lisa).
parent(marge, maggie).
parent(selma, ling).

spouse(abraham, mona).
spouse(clancy, jackie).
spouse(homer, marge).
spouse(marge, homer). 

father(X, Y) :- male(X), parent(X, Y).
mother(X, Y) :- female(X), parent(X, Y).

grandmother(X, Y) :- female(X), parent(X, Z), parent(Z, Y).
grandfather(X, Y) :- male(X), parent(X, Z), parent(Z, Y).

sibling(X, Y) :- parent(Z, X), parent(Z, Y), X \= Y.
brother(X, Y) :- male(X), sibling(X, Y).
sister(X, Y) :- female(X), sibling(X, Y).

aunt(X, Y) :- female(X), sibling(X, Z), parent(Z, Y).
uncle(X, Y) :- male(X), sibling(X, Z), parent(Z, Y).

nephew(X, Y) :- male(X), parent(Z, X), sibling(Z, Y).
niece(X, Y) :- female(X), parent(Z, X), sibling(Z, Y).

mother_in_law(X, Y) :- female(X), parent(X, Z), spouse(Z, Y).
father_in_law(X, Y) :- male(X), parent(X, Z), spouse(Z, Y).
