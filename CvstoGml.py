# Final adjustment to handle the correct column names and delimiters
def csv_to_gml_final(nodes_csv, edges_csv, gml_file):
    try:
        # Create an empty graph
        G = nx.DiGraph()

        # Load nodes from the nodes CSV (adjusting for non-standard delimiter)
        with open(nodes_csv, 'r') as nodes_file:
            reader = csv.reader(nodes_file, delimiter=';')
            headers = next(reader)

            for row in reader:
                node_id = row[headers.index('Id')]
                attributes = {header: row[idx] for idx, header in enumerate(headers) if header != 'Id'}
                
                # Add node with attributes
                G.add_node(node_id, **attributes)

        # Load edges from the edges CSV (adjusting for non-standard delimiter)
        with open(edges_csv, 'r') as edges_file:
            reader = csv.reader(edges_file, delimiter=';')
            headers = next(reader)

            # Check if headers are valid
            if 'Source' not in headers or 'Target' not in headers:
                raise ValueError("Edges CSV must contain 'Source' and 'Target' columns")

            for row in reader:
                source = row[headers.index('Source')]
                target = row[headers.index('Target')]
                edge_attributes = {header: row[idx] for idx, header in enumerate(headers) if header not in ['Source', 'Target']}

                # Convert weight to float if it exists
                if 'Weight' in edge_attributes:
                    edge_attributes['Weight'] = float(edge_attributes['Weight'])

                # Add edge with attributes
                G.add_edge(source, target, **edge_attributes)

        # Write the graph to GML format
        nx.write_gml(G, gml_file)
        print(f"Graph successfully converted to {gml_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Convert the CSVs to GML with the finalized function
csv_to_gml_final(nodes_csv_path, edges_csv_path, output_gml_path)
