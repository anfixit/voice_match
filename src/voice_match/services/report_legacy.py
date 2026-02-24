import numpy as np
import datetime
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import io
import base64


class ForensicReport:
    """Класс для создания экспертного заключения для суда."""

    def __init__(self, file1: str, file2: str):
        """
        Инициализирует отчет для заданных файлов.

        Args:
            file1: Путь к первому аудиофайлу
            file2: Путь ко второму аудиофайлу
        """
        self.file1 = file1
        self.file2 = file2
        self.case_number = f"VM-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.date = datetime.datetime.now().strftime("%d.%m.%Y")
        self.results = {}
        self.confidence_intervals = {}
        self.modification_detected = False
        self.modification_details = {}
        self.final_score = 0.0
        self.consistency = 0.0
        self.verdict = ""
        self.visualizations = {}

    def add_results(self, method: str, values: List[float], weight: float = 1.0):
        """
        Добавляет результаты сравнения для определенного метода.

        Args:
            method: Название метода анализа
            values: Список значений сходства (от 0 до 1)
            weight: Весовой коэффициент метода
        """
        self.results[method] = {
            "values": values,
            "median": np.median(values) if values else 0,
            "mean": np.mean(values) if values else 0,
            "std": np.std(values) if values else 0,
            "min": np.min(values) if values else 0,
            "max": np.max(values) if values else 0,
            "high_similarity_count": sum(1 for x in values if x > 0.9),
            "count": len(values),
            "weight": weight
        }

        # Расчет доверительного интервала (95%)
        if len(values) >= 2:
            import scipy.stats
            mean = np.mean(values)
            std_err = np.std(values, ddof=1) / np.sqrt(len(values))
            ci_low, ci_high = scipy.stats.t.interval(
                0.95, len(values) - 1, loc=mean, scale=std_err
            )
            self.confidence_intervals[method] = (max(0, ci_low), min(1, ci_high))
        else:
            self.confidence_intervals[method] = (0, 1)

    def set_modification_info(self, detected: bool, details: Dict[str, Any]):
        """
        Устанавливает информацию об обнаруженных модификациях голоса.

        Args:
            detected: Флаг обнаружения модификации
            details: Подробная информация о модификациях
        """
        self.modification_detected = detected
        self.modification_details = details

    def set_final_score(self, score: float, consistency: float):
        """
        Устанавливает итоговую оценку и консистентность результатов.

        Args:
            score: Итоговая взвешенная оценка сходства
            consistency: Показатель согласованности различных методов
        """
        self.final_score = score
        self.consistency = consistency

    def set_verdict(self, verdict: str):
        """
        Устанавливает экспертное заключение.

        Args:
            verdict: Текст заключения
        """
        self.verdict = verdict

    def add_visualization(self, name: str, image_data: str):
        """
        Добавляет визуализацию в отчет.

        Args:
            name: Название визуализации
            image_data: Данные изображения в формате base64
        """
        self.visualizations[name] = image_data

    def generate_html_report(self) -> str:
        """
        Генерирует HTML-версию экспертного заключения.

        Returns:
            HTML-код отчета
        """
        # Заголовок отчета
        html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <title>Экспертное заключение №{self.case_number}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                        .header {{ text-align: center; margin-bottom: 30px; }}
                        .section {{ margin-bottom: 20px; }}
                        .section-title {{ background-color: #f0f0f0; padding: 10px; }}
                        table {{ width: 100%; border-collapse: collapse; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f0f0f0; }}
                        .footer {{ margin-top: 30px; border-top: 1px solid #ddd; padding-top: 10px; }}
                        .warning {{ background-color: #fff3cd; border: 1px solid #ffeeba; padding: 10px; margin: 10px 0; }}
                        .success {{ background-color: #d4edda; border: 1px solid #c3e6cb; padding: 10px; margin: 10px 0; }}
                        .info {{ background-color: #d1ecf1; border: 1px solid #bee5eb; padding: 10px; margin: 10px 0; }}
                        .danger {{ background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 10px; margin: 10px 0; }}
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>Экспертное фоноскопическое заключение</h1>
                        <h2>№{self.case_number} от {self.date}</h2>
                    </div>

                    <div class="section">
                        <h3 class="section-title">1. Объекты исследования</h3>
                        <table>
                            <tr>
                                <th>№</th>
                                <th>Файл</th>
                            </tr>
                            <tr>
                                <td>1</td>
                                <td>{self.file1}</td>
                            </tr>
                            <tr>
                                <td>2</td>
                                <td>{self.file2}</td>
                            </tr>
                        </table>
                    </div>
                """

        # Секция обнаружения модификаций
        html += """
                    <div class="section">
                        <h3 class="section-title">2. Проверка подлинности записей</h3>
                """

        if self.modification_detected:
            html += f"""
                        <div class="warning">
                            <h4>⚠️ Обнаружены признаки модификации голоса!</h4>
                            <ul>
                    """

            for file_num, details in self.modification_details.items():
                if details["detected"]:
                    html += f"""
                                <li>Файл {file_num}: Обнаружена модификация типа "{details["type"]}" 
                                с вероятностью {details["confidence"]:.1%}</li>
                            """

            html += """
                            </ul>
                            <p>Данный факт может повлиять на достоверность результатов сравнения.</p>
                        </div>
                    """
        else:
            html += """
                        <div class="success">
                            <p>✅ Признаков искусственной модификации голоса не обнаружено.</p>
                        </div>
                    """

        html += """
                    </div>
                """

        # Секция результатов анализа
        html += """
                    <div class="section">
                        <h3 class="section-title">3. Результаты инструментального анализа</h3>
                        <table>
                            <tr>
                                <th>Метод анализа</th>
                                <th>Медиана</th>
                                <th>95% CI</th>
                                <th>Совпадений &gt;0.9</th>
                                <th>Вес</th>
                            </tr>
                """

        # Заполняем таблицу результатов
        for method, results in self.results.items():
            ci = self.confidence_intervals.get(method, (0, 1))
            ci_width = ci[1] - ci[0]
            ci_class = "success" if ci_width < 0.1 else "info" if ci_width < 0.2 else "warning"

            html += f"""
                        <tr>
                            <td>{method}</td>
                            <td>{results["median"]:.3f}</td>
                            <td class="{ci_class}">{ci[0]:.2f} - {ci[1]:.2f}</td>
                            <td>{results["high_similarity_count"]}/{results["count"]}</td>
                            <td>{results["weight"]:.1f}</td>
                        </tr>
                    """

        html += """
                        </table>

                        <div class="info" style="margin-top: 20px;">
                            <p>Доверительный интервал (CI) показывает диапазон, в котором с вероятностью 95% 
                            находится истинное значение сходства для данного метода.</p>
                        </div>
                    </div>
                """

        # Секция итоговой оценки
        verdict_class = ""
        if self.final_score >= 0.95:
            verdict_class = "success"
        elif self.final_score >= 0.88:
            verdict_class = "info"
        elif self.final_score >= 0.80:
            verdict_class = "info"
        elif self.final_score >= 0.70:
            verdict_class = "warning"
        else:
            verdict_class = "danger"

        html += f"""
                    <div class="section">
                        <h3 class="section-title">4. Итоговая оценка</h3>
                        <div class="{verdict_class}">
                            <p><strong>Взвешенная оценка сходства: {self.final_score:.3f}</strong></p>
                            <p><strong>Доверительный интервал: {self.final_score:.2f} ± {self.consistency:.2f}</strong></p>
                            <p><strong>Консистентность методов: {self.consistency:.3f}</strong> 
                            {"- Высокая согласованность" if self.consistency < 0.1 else "- Средняя согласованность" if self.consistency < 0.15 else "- Низкая согласованность"}</p>
                        </div>
                    </div>

                    <div class="section">
                        <h3 class="section-title">5. Экспертное заключение</h3>
                        <div class="{verdict_class}" style="font-size: 1.2em; padding: 20px;">
                            <p><strong>{self.verdict}</strong></p>
                        </div>

                        <div class="info" style="margin-top: 20px;">
                            <h4>Интерпретация результатов:</h4>
                            <ul>
                                <li>Значение ≥ 0.95: Голоса с высочайшей вероятностью принадлежат одному человеку</li>
                                <li>Значение ≥ 0.88: Голоса с высокой вероятностью принадлежат одному человеку</li>
                                <li>Значение ≥ 0.80: Голоса, вероятно, принадлежат одному человеку</li>
                                <li>Значение ≥ 0.70: Имеется некоторое сходство голосов</li>
                                <li>Значение < 0.70: Голоса вероятно принадлежат разным людям</li>
                            </ul>
                        </div>
                    </div>
                """

        # Секция визуализаций
        if self.visualizations:
            html += """
                        <div class="section">
                            <h3 class="section-title">6. Визуализация результатов</h3>
                    """

            for name, image_data in self.visualizations.items():
                html += f"""
                            <div style="margin-bottom: 30px;">
                                <h4>{name}</h4>
                                <img src="data:image/png;base64,{image_data}" style="max-width:100%">
                            </div>
                        """

            html += """
                        </div>
                    """

        # Подвал отчета
        html += f"""
                    <div class="footer">
                        <p>Отчет сгенерирован автоматически системой voice_match v2.0</p>
                        <p>Дата и время: {datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")}</p>
                    </div>
                </body>
                </html>
                """

        return html
