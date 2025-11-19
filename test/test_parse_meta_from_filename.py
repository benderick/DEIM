from rich import print
from typing import Dict

def parse_meta_from_filename(filename: str) -> Dict:
        """
        从文件名解析元信息
        
        格式: {location}_{time}_{altitude}_{angle}_frame_{frame_num}.jpg
        例如: chenhuachengpark_day_30m_30c_frame_750.jpg
        
        Returns:
            meta: {
                'time': 'day' or 'night',
                'altitude': 30/60/100,
                'angle': 30/90,
                'location': str,
                'frame': int,
                'filename': str
            }
        """
        # 去掉扩展名
        name = filename.replace('.jpg', '').replace('.png', '')
        parts = name.split('_')
        
        meta = {
            'time': None,
            'altitude': None,
            'angle': None,
            'location': None,
            'frame': None,
            'filename': filename
        }
        
        # 解析逻辑
        location_parts = []
        for i, part in enumerate(parts):
            if part in ['day', 'night']:
                meta['time'] = part
                meta['location'] = '_'.join(location_parts) if location_parts else 'unknown'
            elif part.endswith('m') and len(part) > 1:
                try:
                    meta['altitude'] = int(part[:-1])
                except:
                    pass
            elif part.endswith('c') and len(part) > 1:
                try:
                    meta['angle'] = int(part[:-1])
                except:
                    pass
            elif part == 'frame' and i + 1 < len(parts):
                try:
                    meta['frame'] = int(parts[i + 1])
                except:
                    pass
            else:
                # 收集地点名称的部分
                if meta['time'] is None:  # 时间信息之前的都是地点
                    location_parts.append(part)
        
        return meta
    
if __name__ == "__main__":
    filename = "day_jinshanshequ_30m_30c_3_frame_4500.jpg"
    meta = parse_meta_from_filename(filename)
    print(meta)