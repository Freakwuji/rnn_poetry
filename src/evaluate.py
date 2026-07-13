"""
古诗词评价系统

多角度评价体系：
1. 自动指标（客观）：Perplexity、BLEU、新词率
2. 规则化评价（结构）：押韵检测、对仗检测、重复度、字数整齐度
3. 人工评价（主观）：通顺度、意境、押韵、创意
4. AI 评价（综合）：用大模型多维度打分

为什么不能只看 Perplexity：
- Perplexity 只衡量模型对文本的"困惑度"
- 低 Perplexity 不代表生成的诗有诗意
- 可能只是学会了常见搭配，没有创造性
- 需要多角度评价才能全面反映生成质量
"""

import os
import re
import json
import math
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from pathlib import Path


# 韵母表（用于押韵检测）
# 简化版：只考虑韵母是否相同
RHYME_GROUPS = {
    'a': ['阿', '啊', '吧', '拉', '沙', '花', '家', '佳', '麻', '牙', '芽'],
    'o': ['波', '多', '歌', '河', '和', '合', '科', '磨', '婆', '坡'],
    'e': ['车', '得', '地', '各', '客', '乐', '色', '特', '则', '泽'],
    'i': ['笔', '地', '弟', '第', '鸡', '机', '急', '级', '极', '计'],
    'u': ['出', '读', '独', '度', '故', '顾', '路', '录', '入', '树'],
    'v': ['女', '去', '许', '鱼', '雨', '语', '玉', '元', '远', '月'],
    'ai': ['白', '才', '开', '来', '台', '外', '在', '白', '猜', '采'],
    'ei': ['被', '飞', '非', '费', '给', '黑', '累', '美', '每', '妹'],
    'ao': ['报', '到', '高', '好', '号', '老', '劳', '毛', '跑', '少'],
    'ou': ['都', '斗', '够', '后', '厚', '口', '楼', '头', '走', '洲'],
    'an': ['半', '办', '般', '班', '产', '单', '但', '干', '赶', '寒'],
    'en': ['本', '分', '根', '很', '门', '人', '认', '什', '问', '真'],
    'ang': ['长', '场', '常', '当', '方', '光', '行', '黄', '王', '忘'],
    'eng': ['成', '程', '灯', '等', '风', '更', '横', '冷', '生', '声'],
    'in': ['今', '近', '进', '金', '林', '亲', '心', '新', '信', '因'],
    'ing': ['并', '定', '经', '精', '惊', '景', '静', '清', '情', '行'],
    'un': ['本', '存', '村', '分', '魂', '论', '门', '人', '文', '问'],
}


def get_rhyme_group(char):
    """获取字的韵母组"""
    for group, chars in RHYME_GROUPS.items():
        if char in chars:
            return group
    return None


class AutomaticEvaluator:
    """
    自动评价指标

    1. Perplexity：困惑度，越低越好
    2. BLEU：和参考文本的相似度
    3. 新词率：生成的新颖词汇比例
    """

    def __init__(self):
        pass

    def calculate_perplexity(self, model, data_loader, device):
        """
        计算 Perplexity

        Perplexity = exp(loss)
        表示模型对每个字的"困惑程度"
        越低越好，说明模型越确定下一个字是什么
        """
        model.eval()
        total_loss = 0
        total_tokens = 0

        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                logits, _ = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

                total_loss += loss.item()
                mask = targets != 0
                total_tokens += mask.sum().item()

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        return perplexity

    def calculate_bleu(self, generated, reference, n=4):
        """
        计算 BLEU 分数

        BLEU 衡量生成文本和参考文本的 n-gram 重叠度
        值范围 [0, 1]，越高越好

        参数:
            generated: 生成的诗句
            reference: 参考诗句
            n: 最大 n-gram

        示例:
            >>> evaluator = AutomaticEvaluator()
            >>> bleu = evaluator.calculate_bleu("床前明月光", "床前明月光")
            >>> print(f"BLEU: {bleu:.3f}")
            BLEU: 1.000
        """
        def get_ngrams(text, n):
            return [text[i:i+n] for i in range(len(text)-n+1)]

        # 计算各阶 BLEU
        scores = []
        for i in range(1, n+1):
            gen_ngrams = get_ngrams(generated, i)
            ref_ngrams = get_ngrams(reference, i)

            if not gen_ngrams:
                scores.append(0)
                continue

            # 计算匹配数
            ref_counter = Counter(ref_ngrams)
            gen_counter = Counter(gen_ngrams)

            matches = 0
            for ngram, count in gen_counter.items():
                matches += min(count, ref_counter.get(ngram, 0))

            precision = matches / len(gen_ngrams) if gen_ngrams else 0
            scores.append(precision)

        # 几何平均
        if min(scores) > 0:
            log_avg = sum(math.log(s) for s in scores) / len(scores)
            bleu = math.exp(log_avg)
        else:
            bleu = 0

        # 简短惩罚
        brevity_penalty = min(1, len(generated) / max(len(reference), 1))

        return bleu * brevity_penalty

    def calculate_batch_bleu(self, generated_list, reference_list, n=4):
        """
        批量计算 BLEU 分数

        参数:
            generated_list: 生成的诗句列表
            reference_list: 参考诗句列表
            n: 最大 n-gram

        返回:
            avg_bleu: 平均 BLEU 分数
            bleu_list: 每个样本的 BLEU 分数
        """
        bleu_list = []
        for gen, ref in zip(generated_list, reference_list):
            bleu = self.calculate_bleu(gen, ref, n)
            bleu_list.append(bleu)

        avg_bleu = sum(bleu_list) / len(bleu_list) if bleu_list else 0
        return avg_bleu, bleu_list

    def calculate_novelty(self, generated, train_texts):
        """
        计算新词率

        衡量生成内容的新颖程度
        和训练集的重复度越低越好
        """
        gen_chars = set(generated)
        train_chars = set()
        for text in train_texts:
            train_chars.update(text)

        # 新字符比例
        new_chars = gen_chars - train_chars
        novelty = len(new_chars) / len(gen_chars) if gen_chars else 0

        return novelty


class RuleBasedEvaluator:
    """
    规则化评价

    不需要大模型，用规则检测诗句质量
    适合快速评估
    """

    def __init__(self):
        pass

    def check_rhyme(self, poem):
        """
        检测押韵

        规则：检查最后一个字的韵母是否相同
        通常押韵的是第 1、2、4 句的最后一个字
        """
        sentences = re.split(r'[，。！？]', poem)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 0.0, "诗句太短"

        # 获取每句最后一个字
        end_chars = [s[-1] for s in sentences if len(s) > 0]

        if len(end_chars) < 2:
            return 0.0, "有效句子不足"

        # 检查韵母
        rhyme_groups = [get_rhyme_group(c) for c in end_chars]

        # 统计最常见的韵母
        group_counter = Counter(g for g in rhyme_groups if g is not None)
        if not group_counter:
            return 0.3, "无法检测韵母"

        most_common_group, count = group_counter.most_common(1)[0]
        rhyme_ratio = count / len(rhyme_groups)

        if rhyme_ratio >= 0.75:
            return 1.0, f"押韵良好（{most_common_group}韵）"
        elif rhyme_ratio >= 0.5:
            return 0.7, f"基本押韵（{most_common_group}韵）"
        elif rhyme_ratio >= 0.25:
            return 0.4, f"部分押韵"
        else:
            return 0.1, "不押韵"

    def check_parallelism(self, poem):
        """
        检测对仗

        规则：检查相邻句子的词性是否对应
        简化版：检查字数是否相同、位置是否对应
        """
        sentences = re.split(r'[，。！？]', poem)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 0.0, "诗句太短"

        # 检查字数
        lengths = [len(s) for s in sentences]

        # 如果所有句子长度相同，得分较高
        if len(set(lengths)) == 1:
            score = 0.8
            reason = f"字数整齐（每句{lengths[0]}字）"
        elif len(set(lengths)) <= 2:
            score = 0.5
            reason = "字数基本整齐"
        else:
            score = 0.2
            reason = "字数不整齐"

        # 检查位置对应（简化：检查相同位置的字是否相似）
        if all(l == lengths[0] for l in lengths):
            position_match = 0
            for i in range(lengths[0]):
                chars = [s[i] for s in sentences if i < len(s)]
                # 检查是否都是名词、动词等（简化：检查是否都是常见字）
                if len(set(chars)) <= 2:
                    position_match += 1

            position_ratio = position_match / lengths[0] if lengths[0] > 0 else 0
            score = score * 0.5 + position_ratio * 0.5

        return score, reason

    def check_repetition(self, poem):
        """
        检测重复度

        衡量生成内容的重复程度
        重复度越低越好
        """
        chars = list(poem.replace('，', '').replace('。', ''))

        if not chars:
            return 0.0, "空诗句"

        # 字符重复率
        char_counter = Counter(chars)
        unique_chars = len(char_counter)
        total_chars = len(chars)

        # 重复字比例
        repeat_ratio = 1 - (unique_chars / total_chars)

        # 2-gram 重复率
        bigrams = [chars[i:i+2] for i in range(len(chars)-1)]
        bigram_counter = Counter(tuple(b) for b in bigrams)
        unique_bigrams = len(bigram_counter)
        total_bigrams = len(bigrams)

        bigram_repeat = 1 - (unique_bigrams / total_bigrams) if total_bigrams > 0 else 0

        # 综合得分
        score = 1 - (repeat_ratio * 0.5 + bigram_repeat * 0.5)

        return score, f"重复度{repeat_ratio:.1%}"

    def check_length(self, poem):
        """
        检测字数整齐度

        古诗通常要求每句字数相同（五言或七言）
        """
        sentences = re.split(r'[，。！？]', poem)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0, "空诗句"

        lengths = [len(s) for s in sentences]

        # 检查是否为标准的五言或七言
        if all(l == 5 for l in lengths):
            return 1.0, f"标准五言（每句5字）"
        elif all(l == 7 for l in lengths):
            return 1.0, f"标准七言（每句7字）"
        elif len(set(lengths)) == 1:
            return 0.8, f"字数整齐（每句{lengths[0]}字）"
        elif len(set(lengths)) <= 2:
            return 0.5, "字数基本整齐"
        else:
            return 0.2, "字数不整齐"

    def check_imagery(self, poem):
        """
        检测意象丰富度

        古诗常用的意象：月、花、水、山、风、雨、雪、云、柳、梅等
        意象越丰富，诗意越浓
        """
        common_imagery = set('月花水山风雨雪云柳梅竹松鹤星霜露烟霞')
        chars = set(poem.replace('，', '').replace('。', ''))

        imagery_chars = chars & common_imagery
        imagery_count = len(imagery_chars)

        if imagery_count >= 4:
            return 1.0, f"意象丰富（{imagery_count}个）"
        elif imagery_count >= 2:
            return 0.7, f"意象一般（{imagery_count}个）"
        elif imagery_count >= 1:
            return 0.4, f"意象较少（{imagery_count}个）"
        else:
            return 0.1, "无意象"

    def evaluate_poem(self, poem):
        """
        综合评价一首诗

        评价维度：
        1. 押韵：韵脚是否相同
        2. 对仗：字数是否整齐，位置是否对应
        3. 重复度：是否有过多重复
        4. 字数整齐度：是否为标准五言或七言
        5. 意象丰富度：是否包含常见古诗意象

        返回:
            total_score: 总分 (0-1)
            details: 各项得分和说明
        """
        rhyme_score, rhyme_reason = self.check_rhyme(poem)
        parallel_score, parallel_reason = self.check_parallelism(poem)
        repeat_score, repeat_reason = self.check_repetition(poem)
        length_score, length_reason = self.check_length(poem)
        imagery_score, imagery_reason = self.check_imagery(poem)

        # 加权平均
        total_score = (
            rhyme_score * 0.25 +       # 押韵
            parallel_score * 0.2 +     # 对仗
            repeat_score * 0.2 +       # 重复度
            length_score * 0.15 +      # 字数整齐度
            imagery_score * 0.2        # 意象丰富度
        )

        details = {
            'rhyme': {'score': rhyme_score, 'reason': rhyme_reason},
            'parallelism': {'score': parallel_score, 'reason': parallel_reason},
            'repetition': {'score': repeat_score, 'reason': repeat_reason},
            'length': {'score': length_score, 'reason': length_reason},
            'imagery': {'score': imagery_score, 'reason': imagery_reason},
        }

        return total_score, details


class AIEvaluator:
    """
    AI 评价接口

    支持两种方式：
    1. 调用外部大模型 API（如 Qwen、GPT 等）
    2. 本地规则化评价（不需要大模型）

    使用方式：
        evaluator = AIEvaluator()
        # 方式1：生成 prompt，调用外部 API
        prompt = evaluator.generate_prompt(poem, start_word)
        # response = call_api(prompt)
        # score = evaluator.parse_response(response)

        # 方式2：本地规则化评价
        score = evaluator.evaluate_locally(poem, start_word)
    """

    def __init__(self, model_name='qwen-7b-chat'):
        self.model_name = model_name
        self.prompt_template = """你是一个古诗词评价专家。请评价以下 AI 生成的古诗。

生成的诗句：{poem}
起始字：{start_word}

请从以下维度打分（1-10分）：
1. 通顺度：诗句是否通顺，语法是否正确
2. 意境：是否有诗意，画面感如何
3. 押韵：是否押韵，韵脚是否合理
4. 创意：是否有新颖的表达

请严格按照以下 JSON 格式输出：
{{
    "通顺度": <分数>,
    "意境": <分数>,
    "押韵": <分数>,
    "创意": <分数>,
    "总分": <平均分>,
    "评语": "<简要评价>"
}}"""

    def generate_prompt(self, poem, start_word=''):
        """生成评价 prompt（用于调用外部 API）"""
        return self.prompt_template.format(
            poem=poem,
            start_word=start_word
        )

    def parse_response(self, response):
        """解析大模型的返回"""
        try:
            # 尝试提取 JSON
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
        except:
            pass

        # 解析失败，返回默认值
        return {
            '通顺度': 5,
            '意境': 5,
            '押韵': 5,
            '创意': 5,
            '总分': 5,
            '评语': '无法解析'
        }

    def evaluate_locally(self, poem, start_word=''):
        """
        本地规则化评价（不需要大模型）

        根据诗句的特征自动打分：
        - 通顺度：基于常见字比例
        - 意境：基于意象词汇
        - 押韵：基于韵脚检测
        - 创意：基于重复度
        """
        # 通顺度：检查常见字比例
        common_chars = set('的一是了不在有我他这中大来上个人到说时要就出会也得着下自之年过发后与说多去行里用道行然')
        chars = list(poem.replace('，', '').replace('。', ''))
        if not chars:
            smoothness = 1
        else:
            common_ratio = sum(1 for c in chars if c in common_chars) / len(chars)
            smoothness = min(10, int(common_ratio * 15))

        # 意境：检查意象词汇
        imagery_words = {'月': 2, '花': 2, '水': 1, '山': 1, '风': 1, '雨': 1, '雪': 1, '云': 1,
                         '春': 2, '秋': 2, '夜': 1, '梦': 2, '愁': 2, '思': 1, '归': 1}
        imagery_score = sum(imagery_words.get(c, 0) for c in chars)
        imagery = min(10, int(imagery_score * 1.5) + 3)

        # 押韵：检查句尾韵母
        sentences = re.split(r'[，。！？]', poem)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) >= 2:
            end_chars = [s[-1] for s in sentences if len(s) > 0]
            rhyme_groups = [get_rhyme_group(c) for c in end_chars]
            if rhyme_groups:
                from collections import Counter
                group_counter = Counter(g for g in rhyme_groups if g is not None)
                if group_counter:
                    most_common_count = group_counter.most_common(1)[0][1]
                    rhyme_ratio = most_common_count / len(rhyme_groups)
                    rhyme = min(10, int(rhyme_ratio * 12))
                else:
                    rhyme = 4
            else:
                rhyme = 4
        else:
            rhyme = 4

        # 创意：基于重复度（重复越少越有创意）
        unique_ratio = len(set(chars)) / len(chars) if chars else 0
        creativity = min(10, int(unique_ratio * 12) + 2)

        # 总分
        total = (smoothness + imagery + rhyme + creativity) / 4

        return {
            '通顺度': smoothness,
            '意境': imagery,
            '押韵': rhyme,
            '创意': creativity,
            '总分': round(total, 1),
            '评语': '本地规则化评价'
        }

    def batch_evaluate(self, poems, start_words, use_api=False):
        """
        批量评价

        参数:
            poems: 生成的诗句列表
            start_words: 对应的起始字
            use_api: 是否使用 API（False 则用本地评价）

        返回:
            评价结果列表
        """
        results = []
        for poem, start in zip(poems, start_words):
            if use_api:
                # 生成 prompt，需要外部调用 API
                prompt = self.generate_prompt(poem, start)
                results.append({'prompt': prompt, 'poem': poem, 'start': start})
            else:
                # 本地评价
                score = self.evaluate_locally(poem, start)
                results.append(score)

        return results


class HumanEvaluator:
    """
    人工评价管理

    生成评价表，收集打分，统计结果
    """

    def __init__(self):
        pass

    def generate_evaluation_table(self, poems, output_file='evaluation_table.md'):
        """
        生成评价表

        参数:
            poems: 要评价的诗句列表，格式 [(模型名, 起始字, 生成结果), ...]
            output_file: 输出文件
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 古诗词生成评价表\n\n")
            f.write("请对以下生成的古诗进行评价，打分 1-5 分。\n\n")
            f.write("| 编号 | 起始字 | 生成结果 | 通顺度 | 意境 | 押韵 | 创意 | 备注 |\n")
            f.write("|------|--------|----------|--------|------|------|------|------|\n")

            for i, (model, start, poem) in enumerate(poems, 1):
                f.write(f"| {i} | {start} | {poem} | ___ | ___ | ___ | ___ |  |\n")

            f.write("\n\n## 评分标准\n\n")
            f.write("| 分数 | 通顺度 | 意境 | 押韵 | 创意 |\n")
            f.write("|------|--------|------|------|------|\n")
            f.write("| 1 | 完全不通顺 | 无诗意 | 完全不押韵 | 完全抄袭 |\n")
            f.write("| 2 | 勉强可读 | 意境弱 | 勉强押韵 | 少量新意 |\n")
            f.write("| 3 | 基本通顺 | 一般 | 基本押韵 | 有一定创意 |\n")
            f.write("| 4 | 通顺 | 较好 | 押韵 | 较有创意 |\n")
            f.write("| 5 | 非常通顺 | 很有诗意 | 韵律优美 | 非常有创意 |\n")

            f.write("\n\n## 注意事项\n\n")
            f.write("1. 请尽量保持客观，不要因为知道模型类型而影响评分\n")
            f.write("2. 如果某首诗完全不通顺，可以给 0 分\n")
            f.write("3. 备注栏可以填写具体的改进建议\n")

        print(f"评价表已生成: {output_file}")

    def calculate_statistics(self, scores):
        """
        计算统计结果

        参数:
            scores: 打分列表，格式 [(通顺度, 意境, 押韵, 创意), ...]

        返回:
            统计结果
        """
        if not scores:
            return None

        scores_array = np.array(scores)

        stats = {
            'count': len(scores),
            'smoothness_avg': float(np.mean(scores_array[:, 0])),
            'imagery_avg': float(np.mean(scores_array[:, 1])),
            'rhyme_avg': float(np.mean(scores_array[:, 2])),
            'creativity_avg': float(np.mean(scores_array[:, 3])),
            'total_avg': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
        }

        return stats


class ComprehensiveEvaluator:
    """
    综合评价系统

    整合所有评价方式，生成完整的评价报告
    """

    def __init__(self):
        self.auto_evaluator = AutomaticEvaluator()
        self.rule_evaluator = RuleBasedEvaluator()
        self.human_evaluator = HumanEvaluator()
        self.ai_evaluator = AIEvaluator()

    def evaluate_model(self, model, vocab, test_loader, generated_poems,
                       train_texts=None, use_ai=False, device=None):
        """
        综合评价一个模型

        参数:
            model: 模型
            vocab: 词表
            test_loader: 测试数据加载器
            generated_poems: 生成的诗句 [(start_word, poem), ...]
            train_texts: 训练集文本（用于计算新词率）
            use_ai: 是否使用 AI 评价

        返回:
            评价结果字典
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        results = {
            'model_name': model.__class__.__name__,
            'metrics': {},
            'rule_based': {},
            'ai_evaluation': {},
        }

        # 1. 自动指标
        print("[评价] 计算自动指标...")

        # Perplexity
        perplexity = self.auto_evaluator.calculate_perplexity(
            model, test_loader, device
        )
        results['metrics']['perplexity'] = perplexity
        print(f"  Perplexity: {perplexity:.2f}")

        # 2. 规则化评价
        print("\n[评价] 规则化评价...")
        rule_scores = []
        for start_word, poem in generated_poems:
            score, details = self.rule_evaluator.evaluate_poem(poem)
            rule_scores.append({
                'start_word': start_word,
                'poem': poem,
                'score': score,
                'details': details
            })

        # 统计平均分
        avg_score = np.mean([r['score'] for r in rule_scores])
        results['rule_based']['average_score'] = float(avg_score)
        results['rule_based']['details'] = rule_scores
        print(f"  平均分: {avg_score:.3f}")

        # 3. AI 评价（可选）
        if use_ai:
            print("\n[评价] AI 评价...")
            ai_scores = []
            for start_word, poem in generated_poems[:5]:  # 只评价前 5 首
                prompt = self.ai_evaluator.generate_prompt(poem, start_word)
                # 这里需要调用大模型 API
                # response = call_llm_api(prompt)
                # score = self.ai_evaluator.parse_response(response)
                # ai_scores.append(score)

            results['ai_evaluation']['scores'] = ai_scores

        return results

    def generate_report(self, results, output_file='evaluation_report.md'):
        """
        生成评价报告
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 古诗词生成模型评价报告\n\n")

            f.write(f"## 模型: {results['model_name']}\n\n")

            # 自动指标
            f.write("## 自动指标\n\n")
            f.write("| 指标 | 值 | 说明 |\n")
            f.write("|------|-----|------|\n")
            perplexity = results['metrics'].get('perplexity', 'N/A')
            f.write(f"| Perplexity | {perplexity:.2f} | 越低越好 |\n")

            # 规则化评价
            f.write("\n## 规则化评价\n\n")
            f.write(f"平均分: {results['rule_based'].get('average_score', 'N/A'):.3f}\n\n")

            f.write("| 起始字 | 生成结果 | 得分 |\n")
            f.write("|--------|----------|------|\n")
            for item in results['rule_based'].get('details', []):
                f.write(f"| {item['start_word']} | {item['poem'][:20]}... | {item['score']:.3f} |\n")

        print(f"\n评价报告已生成: {output_file}")


# 测试
if __name__ == '__main__':
    # 测试规则化评价
    evaluator = RuleBasedEvaluator()

    test_poems = [
        "春江水暖鸭先知，蒌蒿满地芦芽短。",
        "明月几时有，把酒问青天。",
        "床前明月光，疑是地上霜。",
        "千山鸟飞绝，万径人踪灭。",
        "白日依山尽，黄河入海流。",
    ]

    print("规则化评价测试:")
    print("=" * 60)

    for poem in test_poems:
        score, details = evaluator.evaluate_poem(poem)
        print(f"\n诗句: {poem}")
        print(f"总分: {score:.3f}")
        for key, value in details.items():
            print(f"  {key}: {value['score']:.2f} - {value['reason']}")
