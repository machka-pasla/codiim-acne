import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Union, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Union, Any

from torch import embedding


class Rag :
    model = None
    embild_data = None
    texts_data = None



       
      


    def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 32,
    show_progress_bar: bool = True
          ) -> np.ndarray:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
          )
        
        return embeddings



    def save_embeddings(
        texts: List[str],
        embeddings: np.ndarray,
        file_path: str
       ) -> None:
        data = {
        'texts': texts,
        'embeddings': embeddings
         }
        print(f"Начинаем сохранение данных в файл: {file_path}")
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            print("Сохранение завершено успешно.")
        except IOError as e:
            print(f"Ошибка при сохранении файла {file_path}: {e}")



    def load_embeddings(file_path: str) -> Union[Dict[str, Union[List[str], np.ndarray]], None]:
    
        if not os.path.exists(file_path):
            print(f"Файл не найден: {file_path}")
            return None

        print(f"Начинаем загрузку данных из файла: {file_path}")
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            print("Загрузка завершена успешно.")
        # Проверка, что в загруженных данных есть нужные ключи
            if 'texts' in data and 'embeddings' in data:
                 return data
            else:
                print(f"Файл {file_path} не содержит ожидаемых ключей ('texts', 'embeddings').")
                return None
        except (IOError, pickle.UnpicklingError) as e:
            print(f"Ошибка при загрузке файла {file_path}: {e}")
            return None
        except Exception as e:
             print(f"Неизвестная ошибка при загрузке файла {file_path}: {e}")
             return None
        

    def load_init (file_path):
        model_name = 'paraphrase-multilingual-mpnet-base-v2'
        global model , embild_data , texts_data
   
        try:
            model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Ошибка при загрузке модели {model_name}: {e}")
        embild_data=Rag.load_embeddings(file_path)
        texts_data=embild_data["texts"]
        embild_data=embild_data["embeddings"]

    def init_rag (file_path,texts):
        global model , embild_data , texts_data
        model_name = 'paraphrase-multilingual-mpnet-base-v2'
   
        try:
            model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Ошибка при загрузке модели {model_name}: {e}")
        texts_data=texts
        embild_data=Rag.encode_texts(texts)
        Rag.save_embeddings(texts,embild_data,file_path)










    def find_top(
    dataset_texts: List[str],
    query_text: str,
    top_k: int = 5
) -> List[Dict[str, Union[str, float]]]:
        dataset_embeddings=embild_data
        try:
            global model

            query_embedding = model.encode(
                 [query_text],
                 convert_to_numpy=True,
                 show_progress_bar=False # Обычно нет нужды в прогресс-баре для одного запроса
            )[0]
        except Exception as e:
             print(f"Ошибка при кодировании запроса: {e}")
             return []

        query_embedding = query_embedding.reshape(1, -1)


        print(f"Вычисляем косинусное сходство с {dataset_embeddings.shape[0]} документами...")
        try:
        
            similarity_scores = cosine_similarity(query_embedding, dataset_embeddings)[0]
        except Exception as e:
             print(f"Ошибка при вычислении сходства: {e}")
             return []
        actual_top_k = min(top_k, len(dataset_texts))
        top_k_indices = np.argsort(similarity_scores)[-actual_top_k:][::-1]

    # Собираем результаты
        results = []
        print(f"Собираем топ {len(top_k_indices)} результатов...")
        for idx in top_k_indices:
            results.append({
                'text': dataset_texts[idx],
                'score': float(similarity_scores[idx]) 
            })

        return results









if __name__ == "__main__":
    text = [
        "огуреческое - человеческое",
        "пападун",
        "pf,jktdibt gbljhfcs"

    ]
    rag=Rag
    rag.init_rag("my_text_embeddings.pkl",)
    