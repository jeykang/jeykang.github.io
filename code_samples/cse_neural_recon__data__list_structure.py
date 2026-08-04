import os

def list_structure(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        
        # Print non-image files
        for f in files:
            if not f.endswith(('.png', '.jpg', '.jpeg')):
                print('{}{}'.format(subindent, f))
            else:
                # Just indicate images exist
                pass
        
        if any(f.endswith('.png') for f in files):
             print(f"{subindent}[...images...]")

# Replace with your actual extracted root path
list_structure(os.getcwd())
