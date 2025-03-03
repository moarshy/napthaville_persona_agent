import json


class MemoryTree:
    """
    Spatial memory for generative agents.
    
    This class provides a hierarchical representation of the game world,
    allowing agents to have knowledge about locations and objects.
    The structure follows: world -> sector -> arena -> game_objects
    """
    
    def __init__(self, spatial_memory):
        """
        Initialize the spatial memory tree.
        
        Args:
            f_saved: Path to saved memory file, or False if creating new
        """
        self.tree = {}
        self.tree = spatial_memory

    def print_tree(self):
        """
        Print the entire memory tree for debugging.
        """
        def _print_tree(tree, depth):
            dash = " >" * depth
            if isinstance(tree, list):
                if tree:
                    print(dash, tree)
                return

            for key, val in tree.items():
                if key:
                    print(dash, key)
                _print_tree(val, depth+1)
        
        _print_tree(self.tree, 0)

    def save(self, out_json):
        """
        Save the memory tree to a JSON file.
        
        Args:
            out_json: File path where memory will be saved
        """
        with open(out_json, "w") as outfile:
            json.dump(self.tree, outfile)

    def get_str_accessible_sectors(self, curr_world):
        """
        Get a string listing all sectors in the current world.
        
        Args:
            curr_world: Name of the current world
            
        Returns:
            Comma-separated string of sector names
        
        Example:
            "downtown, residential area, park"
        """
        x = ", ".join(list(self.tree[curr_world].keys()))
        return x

    def get_str_accessible_sector_arenas(self, sector):
        """
        Get a string listing all arenas in the specified sector.
        
        Args:
            sector: Sector address in format "world:sector"
            
        Returns:
            Comma-separated string of arena names
        
        Example:
            "bedroom, kitchen, dining room, office, bathroom"
        """
        curr_world, curr_sector = sector.split(":")
        if not curr_sector:
            return ""
        
        x = ", ".join(list(self.tree[curr_world][curr_sector].keys()))
        return x

    def get_str_accessible_arena_game_objects(self, arena):
        """
        Get a string listing all game objects in the specified arena.
        
        Args:
            arena: Arena address in format "world:sector:arena"
            
        Returns:
            Comma-separated string of object names
        
        Example:
            "phone, charger, bed, nightstand"
        """
        curr_world, curr_sector, curr_arena = arena.split(":")

        if not curr_arena:
            return ""

        try:
            x = ", ".join(list(self.tree[curr_world][curr_sector][curr_arena]))
        except:
            # Fall back to lowercase arena name if case mismatch
            x = ", ".join(list(self.tree[curr_world][curr_sector][curr_arena.lower()]))
        return x


if __name__ == "__main__":
    from pathlib import Path
    import sys
    from io import StringIO
    import os

    repo_path = Path(__file__).parent.parent.parent.parent / "test_data" / "July1_the_ville_isabella_maria_klaus-step-3-19" / "personas" / "Isabella Rodriguez" / "bootstrap_memory" / "spatial_memory.json"

    # make a memory tree from the file
    memory_tree = MemoryTree(repo_path)
    memory_tree.print_tree()

    class TestMemoryTree:
        """Test suite for the MemoryTree class"""
        
        def __init__(self):
            self.test_file = "test_memory_tree.json"
            self.create_test_data()
        
        def create_test_data(self):
            """Create test data for the memory tree"""
            test_data = {
                "oak_hill": {
                    "downtown": {
                        "coffee_shop": ["counter", "tables", "chairs", "coffee_machine"],
                        "bookstore": ["bookshelves", "checkout", "reading_area"],
                        "park": ["benches", "fountain", "playground"]
                    },
                    "residential": {
                        "house_1": ["living_room", "kitchen", "bedroom", "bathroom"],
                        "apartment_building": ["lobby", "elevator", "hallway"],
                        "garden": ["flowers", "bench", "pond"]
                    }
                },
                "maple_valley": {
                    "university": {
                        "library": ["books", "computers", "study_desks"],
                        "cafeteria": ["tables", "food_counter", "vending_machines"],
                        "lecture_hall": ["seats", "podium", "projector"]
                    },
                    "shopping_district": {
                        "clothing_store": ["racks", "fitting_rooms", "register"],
                        "grocery_store": ["produce_section", "checkout", "freezer_aisle"],
                        "electronics_shop": ["display_cases", "service_desk", "accessories"]
                    }
                }
            }
            
            with open(self.test_file, "w") as f:
                json.dump(test_data, f, indent=2)
            
            print(f"Created test data file: {self.test_file}")
        
        def capture_print(self, func):
            """Capture the output of print statements in a function"""
            captured_output = StringIO()
            sys.stdout = captured_output
            func()
            sys.stdout = sys.__stdout__
            return captured_output.getvalue()
        
        def test_initialization(self):
            """Test initialization of the MemoryTree"""
            print("\n=== Testing Initialization ===")
            memory_tree = MemoryTree(self.test_file)
            print("Memory tree initialized successfully from test file")
            
            # Check if the tree structure is as expected
            assert "oak_hill" in memory_tree.tree, "World 'oak_hill' not found in tree"
            assert "downtown" in memory_tree.tree["oak_hill"], "Sector 'downtown' not found in 'oak_hill'"
            assert "coffee_shop" in memory_tree.tree["oak_hill"]["downtown"], "Arena 'coffee_shop' not found in 'downtown'"
            
            print("Structure verification passed")
            return memory_tree
        
        def test_print_tree(self, memory_tree):
            """Test the print_tree method"""
            print("\n=== Testing print_tree Method ===")
            output = self.capture_print(memory_tree.print_tree)
            print("Print tree output captured successfully")
            
            # Check if major elements are in the output
            assert "oak_hill" in output, "World 'oak_hill' not found in print output"
            assert "downtown" in output, "Sector 'downtown' not found in print output"
            assert "coffee_shop" in output, "Arena 'coffee_shop' not found in print output"
            assert "counter" in output, "Game object 'counter' not found in print output"
            
            print("Print tree verification passed")
        
        def test_get_str_accessible_sectors(self, memory_tree):
            """Test getting accessible sectors as a string"""
            print("\n=== Testing get_str_accessible_sectors Method ===")
            
            oak_hill_sectors = memory_tree.get_str_accessible_sectors("oak_hill")
            print(f"Accessible sectors in oak_hill: {oak_hill_sectors}")
            
            # Check if the returned string contains the expected sectors
            assert "downtown" in oak_hill_sectors, "Sector 'downtown' not found in accessible sectors"
            assert "residential" in oak_hill_sectors, "Sector 'residential' not found in accessible sectors"
            
            maple_valley_sectors = memory_tree.get_str_accessible_sectors("maple_valley")
            print(f"Accessible sectors in maple_valley: {maple_valley_sectors}")
            
            # Check if the returned string contains the expected sectors
            assert "university" in maple_valley_sectors, "Sector 'university' not found in accessible sectors"
            assert "shopping_district" in maple_valley_sectors, "Sector 'shopping_district' not found in accessible sectors"
            
            print("get_str_accessible_sectors verification passed")
        
        def test_get_str_accessible_sector_arenas(self, memory_tree):
            """Test getting accessible arenas in a sector as a string"""
            print("\n=== Testing get_str_accessible_sector_arenas Method ===")
            
            downtown_arenas = memory_tree.get_str_accessible_sector_arenas("oak_hill:downtown")
            print(f"Accessible arenas in oak_hill:downtown: {downtown_arenas}")
            
            # Check if the returned string contains the expected arenas
            assert "coffee_shop" in downtown_arenas, "Arena 'coffee_shop' not found in accessible arenas"
            assert "bookstore" in downtown_arenas, "Arena 'bookstore' not found in accessible arenas"
            assert "park" in downtown_arenas, "Arena 'park' not found in accessible arenas"
            
            university_arenas = memory_tree.get_str_accessible_sector_arenas("maple_valley:university")
            print(f"Accessible arenas in maple_valley:university: {university_arenas}")
            
            # Check if the returned string contains the expected arenas
            assert "library" in university_arenas, "Arena 'library' not found in accessible arenas"
            assert "cafeteria" in university_arenas, "Arena 'cafeteria' not found in accessible arenas"
            assert "lecture_hall" in university_arenas, "Arena 'lecture_hall' not found in accessible arenas"
            
            # Test with empty sector
            empty_arenas = memory_tree.get_str_accessible_sector_arenas("oak_hill:")
            assert empty_arenas == "", "Empty sector should return empty string"
            
            print("get_str_accessible_sector_arenas verification passed")
        
        def test_get_str_accessible_arena_game_objects(self, memory_tree):
            """Test getting accessible game objects in an arena as a string"""
            print("\n=== Testing get_str_accessible_arena_game_objects Method ===")
            
            coffee_shop_objects = memory_tree.get_str_accessible_arena_game_objects("oak_hill:downtown:coffee_shop")
            print(f"Accessible objects in oak_hill:downtown:coffee_shop: {coffee_shop_objects}")
            
            # Check if the returned string contains the expected objects
            assert "counter" in coffee_shop_objects, "Object 'counter' not found in accessible objects"
            assert "tables" in coffee_shop_objects, "Object 'tables' not found in accessible objects"
            assert "chairs" in coffee_shop_objects, "Object 'chairs' not found in accessible objects"
            assert "coffee_machine" in coffee_shop_objects, "Object 'coffee_machine' not found in accessible objects"
            
            library_objects = memory_tree.get_str_accessible_arena_game_objects("maple_valley:university:library")
            print(f"Accessible objects in maple_valley:university:library: {library_objects}")
            
            # Check if the returned string contains the expected objects
            assert "books" in library_objects, "Object 'books' not found in accessible objects"
            assert "computers" in library_objects, "Object 'computers' not found in accessible objects"
            assert "study_desks" in library_objects, "Object 'study_desks' not found in accessible objects"
            
            # Test with empty arena
            empty_objects = memory_tree.get_str_accessible_arena_game_objects("oak_hill:downtown:")
            assert empty_objects == "", "Empty arena should return empty string"
            
            print("get_str_accessible_arena_game_objects verification passed")
        
        def test_save(self, memory_tree):
            """Test saving the memory tree to a file"""
            print("\n=== Testing save Method ===")
            
            save_file = "saved_memory_tree.json"
            memory_tree.save(save_file)
            print(f"Memory tree saved to: {save_file}")
            
            # Check if the file exists
            assert os.path.exists(save_file), f"Save file {save_file} not created"
            
            # Load the saved file and check if it has the same structure
            with open(save_file, "r") as f:
                saved_data = json.load(f)
            
            assert "oak_hill" in saved_data, "World 'oak_hill' not found in saved data"
            assert "downtown" in saved_data["oak_hill"], "Sector 'downtown' not found in saved data"
            assert "coffee_shop" in saved_data["oak_hill"]["downtown"], "Arena 'coffee_shop' not found in saved data"
            
            # Clean up
            os.remove(save_file)
            print(f"Removed test file: {save_file}")
            
            print("save method verification passed")
        
        def run_all_tests(self):
            """Run all tests"""
            try:
                memory_tree = self.test_initialization()
                self.test_print_tree(memory_tree)
                self.test_get_str_accessible_sectors(memory_tree)
                self.test_get_str_accessible_sector_arenas(memory_tree)
                self.test_get_str_accessible_arena_game_objects(memory_tree)
                self.test_save(memory_tree)
                
                print("\n=== All Tests Passed Successfully ===")
                
                # Clean up
                os.remove(self.test_file)
                print(f"Removed test file: {self.test_file}")
                
            except AssertionError as e:
                print(f"Test failed: {e}")
            except Exception as e:
                print(f"Error during testing: {e}")
    
    test_memory_tree = TestMemoryTree()
    test_memory_tree.run_all_tests()
