import logging


def right_size(scenes: [str], min_length=500, max_length=4000, delimiter="\n") -> [str]:
    chunks = []
    chunk = ""
    for scene in scenes:
        scene = chunk + delimiter + scene if len(chunk) > 0 else scene
        if len(scene) < min_length:
            chunk = scene
        elif len(scene) > max_length:
            logging.warning('Splitting up overly-large scene with %s chars.', len(scene))
            split_scene = scene.split(delimiter)
            chunksize = int(len(split_scene)/(int(len(scene) / max_length) + 1))
            split_scenes = [delimiter.join(split_scene[i:i + chunksize]) for i in range(0, len(split_scene), chunksize)]
            chunks.extend(split_scenes)
            chunk = ''
        else:
            chunks.append(scene)
            chunk = ''
    return chunks
