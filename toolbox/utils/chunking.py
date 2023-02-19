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
            chunk_size = int(len(split_scene)/(int(len(scene) / max_length) + 1))
            if len(split_scene) == 1 or chunk_size == 0:
                delimiter = '. '
                split_scene = scene.split(delimiter)
                chunk_size = int(len(split_scene)/(int(len(scene) / max_length) + 1))
            assert chunk_size != 0, 'Invalid chunking.'
            split_scenes = [delimiter.join(split_scene[i:i + chunk_size])
                            for i in range(0, len(split_scene), chunk_size)]
            chunks.extend(split_scenes)
            chunk = ''
        else:
            chunks.append(scene)
            chunk = ''
    return chunks
