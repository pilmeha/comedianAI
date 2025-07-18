import random

def split_jokes_file(input_file, output_file):
    # Читаем все строки из исходного файла
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Определяем середину
    section = len(lines) // 2
    
    # Записываем первую половину в первый файл
    with open(output_file, 'w', encoding='utf-8') as f1:
        f1.write('\n'.join(lines[:section]))
    
    # # Записываем вторую половину во второй файл
    # with open(output_file2, 'w', encoding='utf-8') as f2:
    #     f2.write('\n'.join(lines[half:]))

def split_randomly(input_file, output_file, ratio=0.16):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    random.shuffle(lines)
    split_idx = int(len(lines) * ratio)
    
    with open(output_file, 'w', encoding='utf-8') as f1:
        f1.write('\n'.join(lines[:split_idx]))
    
    # with open(output_file2, 'w', encoding='utf-8') as f2:
    #     f2.write('\n'.join(lines[split_idx:]))

# Использование
input_file = 'datasets\\jokes_rand.txt'
output_file = 'datasets\\jokes_dataset_v4.txt'
# output_file2 = 'jokes_part2.txt'

# split_jokes_file(input_file, output_file)

# Использование
# input_file = 'datasets\\translated_filtered_more0_jokes.txt'
# output_file = 'datasets\\jokes_rand.txt'
# output_file2 = 'jokes_part2.txt'

split_randomly(input_file, output_file)

print(f"Файл разделен пополам:")
print(f"- Первая половина: {output_file}")
# print(f"- Вторая половина: {output_file2}")