import re

# Словарь замены тем
theme_dict = {
    "pro-sudey": "судья",
    "pro-studentov": "студент",
    "pro-sport-pro-futbol": "спорт",
    "pro-sisadminov": "сисадмин",
    "pro-semyu": "семья",
    "pro-poruchika-rgevskogo": "Ржевский",
    "pro-putina": "Путин",
    "pro-programmistov": "программист",
    "pro-novih-russkih": "новый русский",
    "pro-narkomanov": "наркоман",
    "pro-mugchin": "мужчина",
    "pro-militsiyu": "милиция",
    "pro-kompyuteri": "компьютер",
    "pro-kino": "кино",
    "pro-inostrantsev": "иностранец",
    "pro-givotnih": "животные",
    "pro-genshin": "женщина",
    "pro-evreev": "евреи",
    "pro-druzey": "друзья",
    "pro-detey": "дети",
    "pro-vovochku": "вовочка",
    "pro-buhgalterov": "бухгалтер",
    "pro-billa-geytsa": "Билл Гейтс",
    "pro-armiu": "армия",
    "pro-alkogolikov": "алкоголик",
    "pro-wow": "ВОВ",
    "poshlie-i-intimnie": "интим",
    "politicheskie": "политика",
    "narodnie": "народ",
    "meditsinskie": "медицина",
    "kriminalnie": "криминал",
    "cherniy-yumor": "черный юмор",
    "tsitati": "цитата",
    "sovetskie": "СССР",
    "skazochnie": "сказка",
    "raznie": "разные",
    "pro-shtirlitsa": "Штирлиц",
    "po-shou-biznes": "бизнес",
    "shkolnie-i-pro-shkolu": "школа",
    "pro-chukchu": "чукча",
    "pro-teshu": "теща",
    "dorognie-pro-dorogu": "дорога",
    "starie-i-borodatie": "старик",
    "aforizmi": "афоризм"
}

def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()  # Читаем все строки
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for line in lines:
            line = line.strip()  # Убираем лишние пробелы и переносы
            if not line:  # Пропускаем пустые строки
                continue
                
            # Находим тему между <|context|> и <|joke|>
            match = re.match(r'(<\|context\|> )(.+?)( <\|joke\|>.+)', line)
            if match:
                prefix = match.group(1)
                old_theme = match.group(2)
                suffix = match.group(3)
                
                # Заменяем тему по словарю
                new_theme = theme_dict.get(old_theme, old_theme)
                
                # Формируем новую строку и добавляем перенос
                new_line = f"{prefix}{new_theme}{suffix}\n"
                outfile.write(new_line)
            else:
                # Если строка не соответствует формату, оставляем как есть + перенос
                outfile.write(f"{line}\n")

# Использование
input_filename = 'output_filtered_more0_jokes.txt'
output_filename = 'translated_filtered_more0_jokes.txt'
process_file(input_filename, output_filename)

print(f"Файл успешно обработан. Результат сохранен в {output_filename}")