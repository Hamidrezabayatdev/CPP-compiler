n cfg_rules[non_terminal]:
            first_alpha = set()
            production_symbols = production.split()
            
            # Compute FIRST(α) for the production
            for symbol in production_symbols:
                if symbol in terminals:
                    first_alpha.add(symbol