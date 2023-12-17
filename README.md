# Direction of Arrival (DoA) estimation alhorithms comparison

## Abstract 

Compararea mai multor metode de DoA. DoA este o tehnica de localizare a directiei din care vine un semnal.
Aceasta este utila in aplicatii precum: radare, sisteme de comunicare, navigatie.

## Running the project

Build the project:
`docker build --platform linux/amd64 -t doa_project:0.0.1 .`
or
`docker compose build`.

Run the project:
`docker run -v ./src:/src doa_project:0.2.1`
or
`docker compose up`.