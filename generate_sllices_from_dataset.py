#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接从 large_piezo_dataset_3x3x3.json 中获取结构数据并生成 SLICES 字符串
这样可以确保 SLICES 编码与实际使用的结构完全一致，避免索引不匹配问题

用法:
    python generate_sllices_from_dataset.py --test  # 测试前5个文件
    python generate_sllices_from_dataset.py --batch  # 处理所有文件
    # 使用 SLICES conda 环境运行
    /d/Anaconda/envs/SLICES/python.exe generate_sllices_from_dataset.py --batch
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow outputs
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"

# Add SLICES to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SLICES-3.1.0', 'SLICES-3.1.0', 'src'))

try:
    from slices.core import SLICES
    from pymatgen.core.structure import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure pymatgen and SLICES are properly installed")
    sys.exit(1)


def extract_symmetry_encoding(slices_string):
    """
    Extract the symmetry group encoding from the beginning of a SLICES string.

    Args:
        slices_string (str): Full SLICES string

    Returns:
        str: Symmetry group encoding part
    """
    tokens = slices_string.split()

    # Symmetry encoding consists of tokens starting with o/+ followed by letters
    # and three-character tokens with O/D/B
    symmetry_tokens = []

    for token in tokens:
        # Check if token is part of symmetry encoding
        if token in ['o', '+'] and len(token) == 1:
            symmetry_tokens.append(token)
        elif token in ['v', 'b', 'g', 'c', 'h'] and len(token) == 1:
            symmetry_tokens.append(token)
        elif len(token) == 3 and all(c in 'OBD' for c in token):
            symmetry_tokens.append(token)
        elif token == 'YBO' or (len(token) == 3 and token[0].isupper() and token[1:].islower()):
            symmetry_tokens.append(token)
        else:
            # Stop when we encounter an element symbol (first letter uppercase)
            if token and token[0].isupper() and (len(token) == 1 or (len(token) == 2 and token[1].islower())):
                break

    return ' '.join(symmetry_tokens)


def process_dataset_entry(entry, backend):
    """
    Process a single dataset entry and generate SLICES string with symmetry info.

    Args:
        entry (dict): Dataset entry from large_piezo_dataset_3x3x3.json
        backend (SLICES): SLICES instance

    Returns:
        dict: Processing results or None if failed
    """
    try:
        # Extract structure information from dataset
        mp_id = entry.get('mp_id', 'unknown')
        structure_dict = entry.get('structure', {})

        # Create pymatgen Structure object
        structure = Structure.from_dict(structure_dict)

        # Get space group info
        analyzer = SpacegroupAnalyzer(structure)
        space_group_num = analyzer.get_space_group_number()
        space_group_symbol = analyzer.get_space_group_symbol()

        # Generate SLICES string with strategy 4 (includes symmetry encoding)
        slices_string = backend.structure2SLICES(structure, strategy=4)

        # Extract symmetry encoding
        symmetry_encoding = extract_symmetry_encoding(slices_string)

        # Get formula
        formula = structure.formula.replace(' ', '')

        return {
            'mp_id': mp_id,
            'formula': formula,
            'space_group_number': space_group_num,
            'space_group_symbol': space_group_symbol,
            'symmetry_encoding': symmetry_encoding,
            'full_slices_string': slices_string,
            'structure_atoms': len(structure),
            'energy_total': entry.get('total', 0),
            'piezo_modulus': entry.get('piezoelectric_modulus', 0),
            'success': True
        }

    except Exception as e:
        mp_id = entry.get('mp_id', 'unknown')
        print(f"[ERROR] Failed to process {mp_id}: {str(e)}")
        return {
            'mp_id': mp_id,
            'error': str(e),
            'success': False
        }


def validate_consistency(dataset_file, slices_file):
    """
    验证生成的SLICES与数据集结构的一致性
    """
    print("[INFO] 验证SLICES与数据集结构的一致性...")

    # 读取数据
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)

    with open(slices_file, 'r') as f:
        slices_data = json.load(f)

    # 创建映射
    slices_by_mp = {entry['mp_id']: entry for entry in slices_data['results'] if entry['success']}

    mismatched_count = 0
    validated_count = 0

    print(f"   数据集样本数: {len(dataset)}")
    print(f"   SLICES成功生成数: {len(slices_by_mp)}")

    # 验证前20个样本
    for entry in dataset[:20]:
        mp_id = entry['mp_id']
        slices_entry = slices_by_mp.get(mp_id)

        if not slices_entry:
            continue

        validated_count += 1

        # 解析SLICES
        slices_string = slices_entry['full_slices_string']
        tokens = slices_string.split()

        # 找到数字部分的开始
        start_idx = 0
        for i, token in enumerate(tokens):
            if token.isdigit():
                start_idx = i
                break

        # 分析边索引
        edge_tokens = tokens[start_idx:]
        indices = []

        for i in range(0, len(edge_tokens), 3):
            if i + 1 < len(edge_tokens):
                try:
                    src = int(edge_tokens[i])
                    dst = int(edge_tokens[i+1])
                    indices.extend([src, dst])
                except ValueError:
                    pass

        # 检查一致性
        dataset_atoms = len(Structure.from_dict(entry['structure']))
        slices_max_idx = max(indices) if indices else 0
        slices_atoms = slices_max_idx + 1

        is_consistent = slices_atoms == dataset_atoms

        if not is_consistent:
            mismatched_count += 1
            print(f"   不匹配: {mp_id}, 数据集: {dataset_atoms}原子, SLICES: {slices_atoms}原子")
        else:
            print(f"   匹配: {mp_id}, {dataset_atoms}原子")

    print(f"\n   验证结果:")
    print(f"     验证样本数: {validated_count}")
    print(f"     匹配样本数: {validated_count - mismatched_count}")
    print(f"     不匹配样本数: {mismatched_count}")
    print(f"   一致性比例: {(validated_count - mismatched_count)/validated_count*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='从数据集生成SLICES字符串')
    parser.add_argument('--test', action='store_true', help='Test on first 5 files')
    parser.add_argument('--batch', action='store_true', help='Process all files')
    parser.add_argument('--output', type=str, default='sllices_from_dataset_2.0.10.json', help='Output JSON file')
    parser.add_argument('--dataset', type=str, default='large_piezo_dataset_3x3x3_fixed.json', help='Dataset file')

    args = parser.parse_args()

    if not args.test and not args.batch:
        print("Please specify either --test or --batch")
        parser.print_help()
        return

    # 检查数据集文件
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        return

    # Initialize SLICES backend
    print("[INFO] 初始化SLICES后端...")
    try:
        # Try chgnet first (more stable)
        backend = SLICES(relax_model="chgnet")
        print("[INFO] 使用CHGNet进行结构优化")
    except:
        try:
            # Fallback to m3gnet
            backend = SLICES(relax_model="m3gnet")
            print("[INFO] 使用M3GNet进行结构优化")
        except Exception as e:
            print(f"[ERROR] 初始化SLICES失败: {e}")
            print("[INFO] 尝试不使用优化模型初始化...")
            try:
                backend = SLICES(relax_model=None)
                print("[INFO] 已初始化SLICES（无优化模型）")
            except Exception as e2:
                print(f"[ERROR] 初始化SLICES失败: {e2}")
                return

    # Load dataset
    print(f"[INFO] 加载数据集: {args.dataset}")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    print(f"[INFO] 找到 {len(dataset)} 个数据集样本")

    # Determine number of files to process
    if args.test:
        dataset = dataset[:5]
        print(f"[INFO] 测试模式，处理 {len(dataset)} 个样本")
    else:
        print(f"[INFO] 批处理模式，处理所有 {len(dataset)} 个样本")

    # Process entries
    results = []
    start_time = time.time()

    for i, entry in enumerate(dataset, 1):
        mp_id = entry.get('mp_id', f'entry_{i}')
        print(f"[{i}/{len(dataset)}] 处理: {mp_id}")

        result = process_dataset_entry(entry, backend)
        results.append(result)

        if result['success']:
            print(f"  [OK] 空间群: {result['space_group_number']} ({result['space_group_symbol']})")
            print(f"  [OK] 原子数: {result['structure_atoms']}")
            print(f"  [OK] SLICES长度: {len(result['full_slices_string'])} 字符")
        else:
            print(f"  [ERROR] 失败: {result.get('error', '未知错误')}")

        # Progress indicator for batch processing
        if args.batch and i % 50 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = (len(dataset) - i) * avg_time
            print(f"\n[PROGRESS] 已处理 {i}/{len(dataset)} 个样本")
            print(f"[PROGRESS] 平均时间: {avg_time:.2f}s/样本")
            print(f"[PROGRESS] 预计剩余时间: {remaining/60:.1f} 分钟\n")

    # Summary
    successful = sum(1 for r in results if r['success'])
    total_time = time.time() - start_time

    print("\n" + "="*60)
    print("汇总")
    print("="*60)
    print(f"总处理样本数: {len(results)}")
    print(f"成功: {successful}")
    print(f"失败: {len(results) - successful}")
    print(f"成功率: {successful/len(results)*100:.1f}%")
    print(f"总时间: {total_time/60:.1f} 分钟")
    print(f"平均时间: {total_time/len(results):.2f} 秒/样本")

    # Save results
    output_data = {
        'generation_info': {
            'timestamp': datetime.now().isoformat(),
            'source_dataset': args.dataset,
            'total_entries': len(results),
            'successful': successful,
            'failed': len(results) - successful,
            'processing_time_seconds': total_time,
            'sllices_version': '2.0.10',
            'strategy': 4,
            'symmetry_encoding': True,
            'consistency_validated': False
        },
        'results': results
    }

    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[INFO] 结果已保存到: {output_path}")

    # Validate consistency
    if successful > 0:
        print("\n" + "="*60)
        print("验证一致性")
        print("="*60)
        validate_consistency(args.dataset, str(output_path))

    # Show examples
    if successful > 0:
        print("\n" + "="*60)
        print("示例结果")
        print("="*60)

        successful_results = [r for r in results if r['success']][:3]
        for result in successful_results:
            print(f"\nMP ID: {result['mp_id']}")
            print(f"化学式: {result['formula']}")
            print(f"空间群: {result['space_group_number']} ({result['space_group_symbol']})")
            print(f"原子数: {result['structure_atoms']}")
            print(f"对称编码: {result['symmetry_encoding']}")
            print(f"SLICES: {result['full_slices_string'][:100]}...")


if __name__ == "__main__":
    main()