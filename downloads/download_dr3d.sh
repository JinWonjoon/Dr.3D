#!/bin/bash

# pretrained model for training
# Bash codes borrowd from https://github.com/KIMGEONUNG/BigColor
mkdir pretrained/ -pv

# 1) Metface
mkdir pretrained/Metface -pv
gdown "1yC5A7xJwZlQWOvU7gFUETSTicwuEL2ga" -O pretrained/Metface/Metface.pkl
gdown "10EbeRlW9qlHoXhibPlUR35VABY3qTzaZ" -O pretrained/Metface/training_options.yaml

# 2) Caricature
mkdir pretrained/Caricature -pv
gdown "1NrzkGO8PUZQWUqoXiaHIDLfP3-HL-COx" -O pretrained/Caricature/Caricature.pkl
gdown "1u1Grlu2QJnlMiBP_n3JSybSH1T41ctxG" -O pretrained/Caricature/training_options.yaml

# 3) Ukiyoe
mkdir pretrained/Ukiyoe -pv
gdown "1xmM5lcwBNiO8CkIBnaHQgOVQST7lO776" -O pretrained/Ukiyoe/Ukiyoe.pkl
gdown "1T7q4y6NUvzvoQj9kx8230STe2_pqGq4W" -O pretrained/Ukiyoe/training_options.yaml

# 4) Anime
mkdir pretrained/Anime -pv
gdown "1WEu6_3hNxgkRtNStPRjStqp63WGM-sVo" -O pretrained/Anime/Anime.pkl
gdown "1EyOkPt4Fq5X2phgclc1dZYoMP15aar7U" -O pretrained/Anime/training_options.yaml

# 5) Webtoon
mkdir pretrained/Webtoon -pv
gdown "1e75e1Lsb8NQ5DpWhQtnppXPZzmIbDqMN" -O pretrained/Webtoon/Webtoon.pkl
gdown "1n3kzWEK01XvoYb-J-_zFGXiU43-jcT_c" -O pretrained/Webtoon/training_options.yaml