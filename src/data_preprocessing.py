import re

def preprocess(sample):
    abstract = sample['abstract'].strip()
    abstract = abstract.replace('\n', '')
    abstract = re.sub(r'[^\w\s]', '', abstract) #remove strange char
    abstract = re.sub(r'\d+', '', abstract) # remove number
    abstract = re.sub(r'\s+', ' ', abstract) # remove extra space
    sample['abstract'] = abstract.lower()
        
    topic = sample['categories'].split(' ')[0]
    sample['categories'] = topic.split('.')[0]
    
    return {
        'text' : sample['title'] + sample['abstract'],
        'label' : sample['categories']
    }
    
def processed_ds(ds, func, num_proc):
    return ds.map(func, num_proc=num_proc)