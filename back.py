import pandas as pd 
import numpy as np
import pickle
import torch 
from PIL import Image
from transformers import AutoModel
import torchvision.models as models
import torch.nn as nn
from torchvision.transforms import CenterCrop, Compose, Pad, Resize, ToTensor
from torchvision import transforms
import Ygpt
from RAG import Rag
from sklearn.neighbors import KNeighborsClassifier
import os



def prediction_final_com(resnet_model, image_arr):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    acne_arr = []
    for i in image_arr:
        img = Image.open(i).convert("RGB")
        img = resize_with_padding(img, (214, 214))
        img_tensor = transform(img).unsqueeze(0)

        resnet_model.eval()
        model1 = resnet_model.to(device)

        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            logits = model1(img_tensor)
            probs = torch.softmax(logits, dim=1)
            print(probs)
            predicted_class = torch.argmax(probs, dim=1).item()
        class_names = ["Comedone", "No Normal"]
        acne_arr.append(class_names[predicted_class])
        
    if acne_arr.count("Comedone") >= 1:
        return "Комедоны есть"
    return "Комедонов нет"


def prediction_final(resnet_model, image_arr):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    acne_arr = []
    for i in image_arr:
        img = Image.open(i).convert("RGB")
        img = resize_with_padding(img, (214, 214))
        img_tensor = transform(img).unsqueeze(0)

        resnet_model.eval()
        model1 = resnet_model.to(device)

        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            logits = model1(img_tensor)
            probs = torch.softmax(logits, dim=1)
            print(probs)
            if abs(probs[0][0] - probs[0][1]) < 0.15:
                predicted_class = 1
            else:
                predicted_class = torch.argmax(probs, dim=1).item()
        class_names = ["Rosacea", "No Normal"]
        acne_arr.append(class_names[predicted_class])
        
    if acne_arr.count("Rosacea") >= 1:
        return "Розацеллы есть"
    return "Розацелл нету"


def resize_with_padding(image, target_size=(214, 120)):
    # Сохраняем соотношение сторон и добавляем отступы (padding), чтобы получить нужный размер
    original_size = image.size
    ratio = min(target_size[0] / original_size[0], target_size[1] / original_size[1])
    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Создаём новое изображение нужного размера и вставляем в центр
    new_img = Image.new("RGB", target_size, (0, 0, 0))  # чёрный фон (можно заменить)
    paste_position = (
        (target_size[0] - new_size[0]) // 2,
        (target_size[1] - new_size[1]) // 2,
    )
    new_img.paste(image, paste_position)
    return new_img

def prediction(resnet_model, image_arr):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    acne_arr = []
    for i in image_arr:
        img = Image.open(i).convert("RGB")
        img = resize_with_padding(img, (214, 214))
        img_tensor = transform(img).unsqueeze(0)

        resnet_model.eval()
        model1 = resnet_model.to(device)

        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            logits = model1(img_tensor)
            probs = torch.softmax(logits, dim=1)
            print(probs)
            predicted_class = torch.argmax(probs, dim=1).item()
            if abs(probs[0][0] - probs[0][1]) < 0.15:
                predicted_class = 2
            elif 0.15 < abs(probs[0][0] - probs[0][1]) < 0.3:
                predicted_class = 0

        class_names = ["Dry", "Normal", "Oily"]
        acne_arr.append(class_names[predicted_class])
        print (acne_arr)
    if acne_arr.count('Normal') > 1:
        return 'Нормальная кожа'
    
    elif acne_arr.count('Dry') > 1:
        return 'Сухая кожа'
    elif acne_arr.count('Oily') > 1:
        return 'Жирная кожа'
    else:
        return 'Нормальная кожа'

def load_mod():
    model = models.resnet18(weights=None)
    num_classes = 3  # укажи своё число классов
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("skin_type_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model



def prediction_acne(resnet_model, image_arr):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    resnet_model.eval()
    acne_arr = []
    for i in image_arr:
        img = Image.open(i).convert("RGB")
        img = resize_with_padding(img, (214, 214))
        img_tensor = transform(img).unsqueeze(0)

        resnet_model.eval()
        model1 = resnet_model.to(device)

        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            logits = model1(img_tensor)
            probs = torch.softmax(logits, dim=1)
            print(probs)
            predicted_class = torch.argmax(probs, dim=1).item()
            if abs(probs[0][0] - probs[0][1]) < 0.2:
                predicted_class = 2
            # elif 0.15 < abs(probs[0][0] - probs[0][1]) < 0.3:
            #     predicted_class = 0

        class_names = ["Acne", "No Acne", "Hesitation"]
        acne_arr.append(class_names[predicted_class])
        
    if acne_arr.count("Acne") >= 1:
        return "Акне есть"
    if acne_arr.count("Hesitation") >= 1:
        return "проблемная кожа (прыщи и тд)"
    return "НЕт акне"

def load_mod_achne():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load("acne_model.pth", map_location=torch.device("cpu")))
    model = model.to(torch.device("cpu"))
    model.eval()
    return model
def load_mod_rossela():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load("rosacea.pth", map_location=torch.device("cpu")))
    model = model.to(torch.device("cpu"))
    model.eval()
    return model
def load_mod_comedon ():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load("comedone_model.pth", map_location=torch.device("cpu")))
    model = model.to(torch.device("cpu"))
    model.eval()
    return model

model_achne=load_mod_achne()
model_skin=load_mod()
model_com=load_mod_comedon()
model_ros=load_mod_rossela()

# Пример изменения в back.py (GET_predict)
def GET_predict (img_arr):
    Ros = prediction_final(model_ros, img_arr)
    Com = prediction_final_com(model_com, img_arr)
    Skin = prediction(model_skin, img_arr)
    Achne = prediction_acne(model_achne, img_arr)
    
    initial_diagnosis_text = f"У пользователя кожа имеет тип: {Skin}\nА акне находятся в состоянии: {Achne}\nРозацеа: {Ros}\nКомедоны: {Com}"
    
    final_recommendation = letllm(initial_diagnosis_text) # letllm принимает текст диагноза

    return {
        "skin_type": Skin,
        "acne_status": Achne,
        "rosacea_status": Ros,
        "comedones_status": Com,
        "initial_diagnosis_summary": initial_diagnosis_text, # или отдельные части, если нужно
        "final_recommendation": final_recommendation
    }



def letllm (res):
    print (res)
    promt = '''
Проанализируй предоставленный ниже список классов связанных с кожанными заболеваний лица и характеристики пользователя. Каждый класс в списке имеет название, и описание в котором содержится инфрмация о симптомах болезни. 
Характеристики пользователя включают четыре параметра: тип кожи, наличие/отсутствие акне, наличие/отсутствие комидонов и наличие/отсутствие розацеи.
Твоя задача — на основе предоставленных характеристик пользователя сопоставить их с описаниями классов и выбрать наиболее подходящий класс.(например , если у пользователя )
В качестве ответа верни только название этого класса, строго в том виде, в котором оно указано в списке. Не добавляй никаких других слов, объяснений или форматирования.
Данные для анализа:
ВАЖНО : при отсутсвии сразу нескольких симптомов у пользователя , лучше отправлять его в классы в которых есть слова (Питают Защищают или Все хорошо)
ВАЖНО : считай что различного рода точки и пятна это тоже самое что акне
ВАЖНО : Считай что прыщи и комидоны ожно и тоже 
ВАЖНО : считай что Разацелла это покраснение и зуд это ожно и тоже
ВАЖНО : Кожа может иметь только одно знаение , так что если кожа например сухая или нормальная то она точно не жирная , тогда те классы где говорится про жирную становятся менее предпочтительными
1. Список классов кожных заболеваний:
запятая является символом разделителем 
         Название кластера        ,                         Описание симптомов
"Увлажнение и Питание (Гиалуроновая кислота, масла, церамиды)","Симптомы: Сухость, шелушение, стянутость кожи, ощущение дискомфорта, обезвоженность, потеря эластичности из-за недостатка влаги. Нарушение водного баланса."
"Антивозрастной уход и Лифтинг (Ретинол, пептиды, коллаген)","Симптомы: Морщины (мимические и статические), потеря упругости и эластичности кожи, изменение овала лица, тусклый цвет лица, возрастная пигментация."
"Уход за проблемной и жирной кожей (Кислоты, себорегуляция, анти-акне)","Симптомы: Излишняя жирность (себорея), акне (прыщи, угри, папулы, пустулы), комедоны (черные точки, белые точки), расширенные поры, воспаления, жирный блеск, постакне (пятна, рубцы). "
"Выравнивание тона, сияние и борьба с пигментацией (Витамин С, ниацинамид, кислоты отшелушивающие)","Симптомы: Пигментные пятна (веснушки, постакне, возрастные пятна, мелазма), неровный тон кожи, тусклый цвет лица, отсутствие здорового сияния. "
"Защита от солнца (SPF)","Симптомы/риски: Склонность к солнечным ожогам, появление или усугубление пигментации под воздействием УФ-лучей, признаки фотостарения (преждевременные морщины, потеря упругости, сухость, утолщение кожи), повышенный риск развития новообразований кожи. "
"Очищение кожи (Энзимы, гидрофильные масла, ПАВ, глины, пилинги)","Симптомы: Загрязненная кожа, остатки макияжа, забитые поры, черные точки, излишки кожного сала (себума), тусклый цвет лица из-за скопления омертвевших клеток, неровная текстура кожи. "
"Успокоение и восстановление барьера (Центелла, алоэ, пантенол, пробиотики)","Симптомы: Покраснение, раздражение, зуд, жжение, шелушение, чувство стянутости, повышенная реактивность на внешние факторы и косметику, признаки нарушения защитного (липидного) барьера кожи (сухость, обезвоженность даже при жирной коже)."
"Увлажнение, восстановление барьера и успокоение кожи (Липиды, керамиды, пантенол)","Симптомы: Комплекс проявлений сухости, обезвоженности и чувствительности – сухость, стянутость, шелушение, покраснение, раздражение, зуд, ослабленный и поврежденный защитный липидный барьер, повышенная чувствительность к внешним раздражителям. "
"Уход за кожей вокруг глаз и патчи","Симптомы: Темные круги под глазами, отечность (мешки под глазами), мелкие морщинки (""гусиные лапки""), сухость, потеря тонуса и эластичности тонкой кожи век, признаки усталости. "
"Многофункциональные и общеукрепляющие средства (Витаминные комплексы, наборы)","Симптомы/потребности: Общая ""усталость"" кожи, тусклый цвет лица, признаки дефицита витаминов и минералов (сухость, шелушение, снижение тонуса, появление высыпаний), потребность в комплексном улучшении состояния кожи, антиоксидантной защите, профилактике старения. [5, 8, 12, 17, 21, 27, 29, 33, 34, 37] Могут включать косметические средства с витаминными комплексами (A, C, E, группы B), антиоксидантами, минералами, а также БАДы для приема внутрь, направленные на улучшение здоровья кожи, волос и ногтей (коллаген, гиалуроновая кислота, омега-3, витамины). [5, 21, 27, 33, 37]"
"Уход за губами","Симптомы: Сухость, трещины, шелушение, обветривание кожи губ, дискомфорт, заеды (ангулярный хейлит), ощущение стянутости, иногда потеря объема. "
"все хорошо","с вашей кожей все хорошо"
'''
    resu = Ygpt.get_drug(promt,res)
    return RAG_1(resu)







def RAG_1 (resu):
    promt = '''
Название кластера  |	Описание кластера
Увлажнение и Питание (Гиалуроновая кислота, масла, церамиды)  |	Продукты, нацеленные на глубокое увлажнение, питание кожи, восстановление водного баланса и смягчение. Часто содержат гиалуроновую кислоту, различные масла, экстракты, церамиды. Подходят для сухой, обезвоженной кожи, а также для поддержания гидробаланса нормальной кожи.
Антивозрастной уход и Лифтинг (Ретинол, пептиды, коллаген)  |	Средства для борьбы с признаками старения: морщинами, потерей упругости и эластичности, изменением овала лица. Ключевые компоненты: ретиноиды, пептиды, коллаген, антиоксиданты, стволовые клетки, факторы роста. Часто имеют регенерирующий и обновляющий эффект.
Уход за проблемной и жирной кожей (Кислоты, себорегуляция, анти-акне)  |	Продукты для кожи, склонной к акне, высыпаниям, черным точкам, излишней жирности и расширенным порам. Содержат кислоты (AHA, BHA, PHA), компоненты, регулирующие выработку себума (цинк, ниацинамид), противовоспалительные и антибактериальные ингредиенты.
Выравнивание тона, сияние и борьба с пигментацией (Витамин С, ниацинамид, кислоты отшелушивающие)  |	Средства, направленные на осветление пигментных пятен различного происхождения, выравнивание общего тона кожи, придание ей здорового сияния и свежести. Часто содержат витамин С, ниацинамид, арбутин, кислоты (AHA, PHA), растительные экстракты с осветляющим действием.
Защита от солнца (SPF)  |	Продукты с SPF-фильтрами для защиты кожи от UVA/UVB излучения, предотвращения фотостарения, пигментации и солнечных ожогов. Могут содержать дополнительные ухаживающие компоненты (увлажнение, антиоксиданты).
Очищение кожи (Энзимы, гидрофильные масла, ПАВ, глины, пилинги)  |	Средства для удаления макияжа, загрязнений, излишков себума и омертвевших клеток. Включают гидрофильные масла, пенки/гели с ПАВ, энзимные пудры, пилинги-скатки, глиняные маски, кислотные пилинги. Могут иметь дополнительные эффекты (сужение пор, матирование, успокоение).
Успокоение и восстановление барьера (Центелла, алоэ, пантенол, пробиотики)  |	Продукты для чувствительной, раздраженной, поврежденной кожи, а также кожи с нарушенным защитным барьером. Содержат успокаивающие (центелла, алоэ, ромашка, овес), восстанавливающие (пантенол, церамиды, липиды) и поддерживающие микробиом (пре/пробиотики) компоненты.
Увлажнение, восстановление барьера и успокоение кожи (Липиды, керамиды, пантенол)  |	Этот кластер объединяет продукты, которые не только интенсивно увлажняют, но и активно работают над восстановлением липидного барьера кожи, что критично для сухой, обезвоженной и чувствительной кожи. Часто содержат керамиды, липиды, пантенол, сквалан, масло ши. Могут также успокаивать раздражения.
Декоративная косметика с ухаживающими компонентами (BB/CC кремы, тональные с уходом, SPF)  |	Продукты, сочетающие маскирующие свойства декоративной косметики (выравнивание тона, скрытие недостатков) с ухаживающими функциями (увлажнение, питание, SPF-защита, антивозрастные компоненты). Включают BB/CC-кремы, тональные основы с сыворотками, консилеры с уходом, праймеры.
Уход за кожей вокруг глаз и патчи  |	Специализированные средства для деликатной зоны вокруг глаз. Направлены на борьбу с темными кругами, отечностью, мелкими морщинками, увлажнение и повышение тонуса. Включают кремы, гели, сыворотки и патчи с активными компонентами (кофеин, пептиды, гиалуроновая кислота, витамины).
Многофункциональные и общеукрепляющие средства (Витаминные комплексы, наборы)  |	Наборы или отдельные продукты, предлагающие комплексный уход или решающие сразу несколько задач (например, увлажнение + антиоксидантная защита + улучшение тона). Часто содержат витаминные комплексы, минералы, растительные экстракты, антиоксиданты. Также сюда отнесены БАДы для кожи.
Уход за губами  |	Бальзамы, блески, масла, предназначенные для увлажнения, питания, защиты и восстановления кожи губ. Могут придавать объем или оттенок. Ключевые компоненты: масла, воски, гиалуроновая кислота, пантенол.
'''
    resu = Ygpt.get_drug(promt,"в пронте у тебя таблица - выбери из нее строку с назвнием "+resu+"и дальше работай используя информацию из нее. кратко Опиши пользователю что у него с кожей а так же какие средства стоит использовать чтобы это исправить (не более 2 предложений)")
    return resu 