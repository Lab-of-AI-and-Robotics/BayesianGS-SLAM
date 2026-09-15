import json
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(
    description='VarSplat: Uncertainty-aware 3DGS SLAM (CVPR 2026)')
parser.add_argument('--input_path', default="./keyframe_count_log.json")

args = parser.parse_args()
file_path = args.input_path

with open(file_path, 'r') as f:
    data = json.load(f)

frame_ids = data['frame_id']
cumulative_counts = data['cumulative_keyframe_count']

baesed_frame_ids  = range(frame_ids[-1])

plt.figure(figsize=(10, 6))

plt.step(frame_ids, cumulative_counts, where='post', color='#1f77b4', linewidth=2, label='Keyframes')
plt.step(baesed_frame_ids, baesed_frame_ids, where='post', color='#ff7f0e', linewidth=2, label='base keyframes', alpha=0.5, linestyle='--')


plt.title('Cumulative Keyframe Count over Time (Frames)', fontsize=14, pad=15)
plt.xlabel('Frame ID', fontsize=12)
plt.ylabel('Cumulative Keyframe Count', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, max(frame_ids) + 50)  # x축 여백 추가
plt.ylim(0, max(frame_ids) + 10) # y축 여백 추가
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig('cumulative_keyframes_plot.pdf', dpi=300, bbox_inches='tight') # 논문용 PDF 저장
plt.show()