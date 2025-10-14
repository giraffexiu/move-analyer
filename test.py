import py_move_analyer

def main():

    project_path = "/Users/giraffe/Downloads/Work/Sui/move-analyer/test/deepbookv3" 
    
    search_query = "new_order_deep_price"  # symbol_name or module::symbol_name
    
    try:

        analyzer = py_move_analyer.analyze_project_advanced(project_path)
        print(f"success")
        
        results = py_move_analyer.symbol_finder(analyzer, search_query)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"{result}")
                if i < len(results):
                    print()  
        else:
            print(f"\nnone")
            
    except Exception as e:
        print(f" {e}")

if __name__ == "__main__":
    main()