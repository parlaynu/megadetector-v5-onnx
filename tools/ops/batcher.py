

def batcher(pipe, batch_size):
    
    items = []
    for item in pipe:
        items.append(item)
        
        if len(items) == batch_size:
            item = _merge_items(items)
            items = []
            yield item

    if len(items) > 0:
        print(f"Warning: not processing final {len(items)} images - smaller than batch size")


def _merge_items(items):
    item = items[0]
    
    for i in items[1:]:
        for k in item.keys():
            item[k].append(i[k][0])

    return item

